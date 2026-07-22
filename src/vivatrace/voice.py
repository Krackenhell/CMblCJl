from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as websocket_connect

from .grammar import grammar_runtime_identity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WHISPER_CLI = PROJECT_ROOT / "tools" / "whisper" / "Release" / "whisper-cli.exe"
WHISPER_MODEL = PROJECT_ROOT / "models" / "ggml-base.en.bin"
VAD_MODEL = PROJECT_ROOT / "models" / "ggml-silero-v6.2.0.bin"
VOICE_MANIFEST = PROJECT_ROOT / "models" / "voice-model-manifest.json"
VOICE_PORT = int(os.getenv("VIVATRACE_VOICE_PORT", "8765"))


_VOICE_META_PATTERNS = (
    r"^(?:are|can|could|do|did|will|would)\s+you\b",
    r"^(?:please\s+)?(?:listen|wait|hold on|let me)\b",
    r"^(?:sorry[, ]+)?(?:are you there|can you hear me)\b",
    r"\b(?:check|listen to|look at)\s+(?:my|this|the)\s+(?:example|answer)\b",
    r"\bhow do you\s+(?:like|find|rate)\b",
    r"\bwhy do you\s+(?:ask|repeat)\b",
    r"^(?:but\s+)?i\s+(?:already\s+)?(?:gave|answered|said)\s+you\b",
)


def is_assessable_spoken_turn(text: str) -> bool:
    """Return False for microphone checks and requests addressed to the tutor."""
    normalized = " ".join(str(text or "").lower().strip().split())
    if len(re.findall(r"[a-z]+(?:'[a-z]+)?", normalized)) < 3:
        return False
    return not any(re.search(pattern, normalized) for pattern in _VOICE_META_PATTERNS)


def voice_runtime_identity() -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    if VOICE_MANIFEST.exists():
        try:
            manifest = json.loads(VOICE_MANIFEST.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (WHISPER_CLI, WHISPER_MODEL, VAD_MODEL)
        if not path.exists()
    ]
    grammar = grammar_runtime_identity()
    missing.extend(grammar["missing"])
    return {
        "ready": not missing,
        "missing": missing,
        "asr": str(manifest.get("asr_model") or "Whisper base.en"),
        "vad": str(manifest.get("vad_model") or "Silero VAD"),
        "tts": str(manifest.get("tts") or "Windows SAPI offline"),
        "grammar": str(manifest.get("grammar_checker") or grammar["name"]),
        "runtime_version": str(manifest.get("runtime_version") or "unknown"),
        "asr_sha256": str(manifest.get("asr_model_sha256") or ""),
        "vad_sha256": str(manifest.get("vad_model_sha256") or ""),
    }


def voice_server_ready(host: str = "127.0.0.1", port: int = VOICE_PORT) -> bool:
    try:
        with websocket_connect(
            f"ws://{host}:{port}", open_timeout=0.5, close_timeout=0.5
        ):
            return True
    except (OSError, TimeoutError, WebSocketException):
        return False


def ensure_voice_server(port: int = VOICE_PORT) -> dict[str, Any]:
    identity = voice_runtime_identity()
    if not identity["ready"]:
        return {**identity, "server_ready": False}
    if voice_server_ready(port=port):
        return {**identity, "server_ready": True, "port": port}

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    stdout = (log_dir / "voice-server.stdout.log").open("a", encoding="utf-8")
    stderr = (log_dir / "voice-server.stderr.log").open("a", encoding="utf-8")
    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "vivatrace.voice_server",
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        creationflags=creation_flags,
    )
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if voice_server_ready(port=port):
            return {**identity, "server_ready": True, "port": port}
        time.sleep(0.25)
    return {**identity, "server_ready": False, "port": port}


def acoustic_fluency_metrics(
    pcm16: bytes,
    transcript: str,
    sample_rate: int = 16_000,
) -> dict[str, float | int]:
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    duration = len(samples) / sample_rate if sample_rate else 0.0
    signal_peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    signal_rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    frame_size = max(int(sample_rate * 0.02), 1)
    frame_count = len(samples) // frame_size
    if frame_count:
        framed = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
        rms = np.sqrt(np.mean(np.square(framed), axis=1) + 1e-12)
        adaptive_threshold = max(0.008, float(np.percentile(rms, 20)) * 2.2)
        voiced_share = float(np.mean(rms >= adaptive_threshold))
    else:
        voiced_share = 0.0
    pause_ratio = 1.0 - voiced_share
    words = [token for token in transcript.replace("—", " ").split() if token.strip(".,!?")]
    word_count = len(words)
    words_per_minute = word_count / duration * 60 if duration > 0 else 0.0
    lowered = f" {transcript.lower()} "
    fillers = sum(
        lowered.count(f" {filler} ")
        for filler in ("um", "uh", "erm", "hmm", "like", "you know")
    )
    if 90 <= words_per_minute <= 170:
        pace_score = 1.0
    elif 65 <= words_per_minute <= 200:
        pace_score = 0.7
    elif word_count >= 3:
        pace_score = 0.4
    else:
        pace_score = 0.1
    pause_score = max(0.0, min(1.0, 1.0 - pause_ratio / 0.70))
    filler_score = max(0.0, 1.0 - fillers / max(word_count / 8, 1))
    fluency_score = 0.45 * pace_score + 0.40 * pause_score + 0.15 * filler_score
    return {
        "duration_seconds": round(duration, 2),
        "signal_peak": round(signal_peak, 4),
        "signal_rms": round(signal_rms, 4),
        "word_count": word_count,
        "words_per_minute": round(words_per_minute, 1),
        "pause_ratio": round(pause_ratio, 3),
        "filler_count": fillers,
        "fluency_score": round(fluency_score, 3),
    }


def overall_speaking_score(assessment: dict[str, Any], metrics: dict[str, Any]) -> float:
    dimensions = [
        float(assessment.get("grammar_score") or 0),
        float(assessment.get("vocabulary_score") or 0),
        float(assessment.get("relevance_score") or 0),
        float(metrics.get("fluency_score") or 0),
    ]
    return round(sum(dimensions) / len(dimensions), 4)


def voice_component_html(config: dict[str, Any]) -> str:
    component_config = dict(config)
    component_config.setdefault(
        "websocket_url", f'ws://127.0.0.1:{int(component_config.get("port") or VOICE_PORT)}'
    )
    safe_config = json.dumps(component_config, ensure_ascii=False).replace("<", "\\u003c")
    template = r"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<style>
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Arial,sans-serif;color:#18231f;background:#f5f6f1}
.shell{border:1px solid #dfe5df;border-radius:22px;background:white;overflow:hidden;box-shadow:0 12px 28px rgba(20,55,43,.08)}
.top{padding:20px 22px;background:linear-gradient(120deg,#123d2f,#1a7053);color:white;display:flex;justify-content:space-between;gap:18px;align-items:center}
.top h2{font-size:20px;margin:0 0 5px}.top p{font-size:13px;margin:0;color:#d7e8e1}.local{font-size:11px;font-weight:700;color:#c9f26b;text-align:right}
.status{display:flex;align-items:center;gap:9px;padding:12px 22px;border-bottom:1px solid #edf0ed;font-size:13px}.dot{width:10px;height:10px;border-radius:50%;background:#b1b8b4}.dot.live{background:#35c58b;box-shadow:0 0 0 6px rgba(53,197,139,.12)}
.meter{height:5px;flex:1;background:#e7ece8;border-radius:10px;overflow:hidden}.meter i{display:block;height:100%;width:0;background:#35c58b;transition:width .08s}
.chat{height:390px;overflow:auto;padding:18px 20px;background:#fafbf8}.msg{max-width:82%;padding:11px 14px;border-radius:16px;margin:0 0 12px;line-height:1.42;font-size:14px;white-space:pre-wrap}.bot{background:#e8f4ef;border-bottom-left-radius:5px}.student{background:#173f31;color:white;margin-left:auto;border-bottom-right-radius:5px}.system{background:#fff4db;color:#654d13;margin-left:auto;margin-right:auto;font-size:12px}.feedback{margin:-6px 0 14px auto;max-width:82%;font-size:12px;color:#53635c;border-left:3px solid #c9f26b;padding:7px 10px;background:#f5f8ee}.controls{padding:15px 20px 18px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.controls button{border:0;border-radius:12px;padding:11px 16px;font-weight:700;cursor:pointer}.primary{background:#176b50;color:white}.secondary{background:#e6f0eb;color:#174d3b}.danger{background:#f5e3df;color:#9c392c}.controls button:disabled{opacity:.45;cursor:not-allowed}.hint{font-size:12px;color:#67766f;flex:1;min-width:220px}.scores{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:0 20px 18px}.score{background:#f3f6f2;border-radius:11px;padding:8px 10px}.score b{display:block;font-size:15px}.score span{font-size:10px;color:#68766f}
</style>
</head>
<body>
<div class="shell">
  <div class="top"><div><h2>Голосовая Viva · __TOPIC__</h2><p>Говорите по-английски. Бот слышит вас даже во время своей реплики.</p></div><div class="local">ПОЛНОСТЬЮ ЛОКАЛЬНО<br>Whisper · Qwen · SAPI</div></div>
  <div class="status"><span id="dot" class="dot"></span><span id="status">Нажмите «Начать диалог»</span><div class="meter"><i id="level"></i></div></div>
  <div id="chat" class="chat"><div class="msg bot">Hi! Let’s have a short B2 speaking practice. Give one original English example from today’s topic.</div></div>
  <div id="scores" class="scores" style="display:none"></div>
  <div class="controls"><button id="start" class="primary">Начать диалог</button><button id="manual" class="secondary" disabled>Начать реплику вручную</button><button id="stop" class="danger" disabled>Завершить диалог</button><div class="hint">Пауза до пяти секунд остаётся внутри одной реплики. Для немедленной отправки используйте ручную кнопку. Зелёная полоска должна двигаться во время речи.</div></div>
</div>
<script>
const CONFIG=__CONFIG__;
const SAMPLE_RATE=16000,SILENCE_MS=5200,MAX_UTTERANCE_MS=45000;
let ws,audioContext,stream,source,processor,zeroGain,speech=false,manualCapture=false,awaitingResponse=false,speechStartedAt=0,lastVoiceAt=0,hotFrames=0,quietFrames=0,noiseFloor=0.008,preRoll=[],currentAudio=null,running=false,botSpeaking=false,lastError='',lastErrorAt=0;
const statusEl=document.getElementById('status'),dot=document.getElementById('dot'),level=document.getElementById('level'),chat=document.getElementById('chat'),startBtn=document.getElementById('start'),manualBtn=document.getElementById('manual'),stopBtn=document.getElementById('stop');
function setStatus(text,live=false){statusEl.textContent=text;dot.className='dot'+(live?' live':'')}
function addMessage(kind,text){const node=document.createElement('div');node.className='msg '+kind;node.textContent=text;chat.appendChild(node);chat.scrollTop=chat.scrollHeight;return node}
function addFeedback(data){if(!data||data.scoring_available===false)return;const node=document.createElement('div');node.className='feedback';const audit=(data.grammar_findings||[]).length?' Проверено: независимое правило + локальная Qwen.':(data.structural_evidence||[]).length?' Проверено: структурный анализ relative clause + локальная Qwen.':' Проверено: локальная Qwen.';node.textContent=data.feedback_ru+(data.correction_en?' Исправление: '+data.correction_en:'')+audit;chat.appendChild(node);chat.scrollTop=chat.scrollHeight;const metrics=data.metrics||{};const scores=[['Грамматика',data.grammar_score],['Словарь',data.vocabulary_score],['Содержание',data.relevance_score],['Беглость',metrics.fluency_score]];const box=document.getElementById('scores');box.innerHTML='';scores.forEach(([name,value])=>{const n=document.createElement('div');n.className='score';n.innerHTML='<b>'+Math.round((value||0)*100)+'%</b><span>'+name+'</span>';box.appendChild(n)});box.style.display='grid'}
function stopPlayback(){if(currentAudio){try{currentAudio.stop()}catch(e){}currentAudio=null}botSpeaking=false}
function downsample(input,inputRate){if(inputRate===SAMPLE_RATE)return input;const ratio=inputRate/SAMPLE_RATE;const length=Math.round(input.length/ratio);const output=new Float32Array(length);let pos=0;for(let i=0;i<length;i++){const next=Math.round((i+1)*ratio);let sum=0,count=0;for(;pos<next&&pos<input.length;pos++){sum+=input[pos];count++}output[i]=count?sum/count:0}return output}
function toPCM16(float32){const out=new Int16Array(float32.length);for(let i=0;i<float32.length;i++){const s=Math.max(-1,Math.min(1,float32[i]));out[i]=s<0?s*32768:s*32767}return out}
function sendJson(data){if(ws&&ws.readyState===WebSocket.OPEN)ws.send(JSON.stringify(data))}
function beginSpeech(manual=false){if(speech||awaitingResponse||!running)return;speech=true;manualCapture=manual;speechStartedAt=Date.now();lastVoiceAt=Date.now();quietFrames=0;if(botSpeaking){stopPlayback();sendJson({type:'interrupt'})}sendJson({type:'speech_start'});if(!manual)preRoll.forEach(chunk=>ws.send(chunk.buffer));preRoll=[];manualBtn.textContent='Отправить реплику';manualBtn.className='primary';setStatus(manual?'Ручная запись идёт. Произнесите фразу и нажмите «Отправить реплику».':'Записываю реплику… Сделайте паузу или нажмите «Отправить реплику».',true)}
function endSpeech(reason='manual'){if(!speech)return;speech=false;manualCapture=false;awaitingResponse=true;hotFrames=0;quietFrames=0;sendJson({type:'speech_end'});manualBtn.disabled=true;manualBtn.textContent='Реплика отправлена';manualBtn.className='secondary';setStatus(reason==='timeout'?'Достигнут лимит записи. Распознаю реплику…':'Распознаю и формирую ответ…',true)}
async function playWave(buffer){if(!running)return;stopPlayback();try{const decoded=await audioContext.decodeAudioData(buffer.slice(0));const node=audioContext.createBufferSource();node.buffer=decoded;node.connect(audioContext.destination);node.onended=()=>{if(currentAudio===node){currentAudio=null;botSpeaking=false;setStatus('Слушаю. Начните говорить.',true)}};currentAudio=node;botSpeaking=true;node.start()}catch(e){addMessage('system','Не удалось воспроизвести локальный голос, текст ответа сохранён.')  }}
async function start(){startBtn.disabled=true;setStatus('Запрашиваю доступ к микрофону…',true);try{stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});audioContext=new (window.AudioContext||window.webkitAudioContext)();await audioContext.resume();source=audioContext.createMediaStreamSource(stream);processor=audioContext.createScriptProcessor(4096,1,1);zeroGain=audioContext.createGain();zeroGain.gain.value=0;source.connect(processor);processor.connect(zeroGain);zeroGain.connect(audioContext.destination);ws=new WebSocket(CONFIG.websocket_url);ws.binaryType='arraybuffer';ws.onopen=()=>{running=true;awaitingResponse=false;manualBtn.disabled=false;stopBtn.disabled=false;sendJson({type:'configure',...CONFIG});setStatus('Слушаю. Начните говорить.',true)};ws.onmessage=async event=>{if(typeof event.data!=='string'){awaitingResponse=false;manualBtn.disabled=false;manualBtn.textContent='Начать реплику вручную';await playWave(event.data);return}const data=JSON.parse(event.data);if(data.type==='transcript')addMessage('student',data.text);if(data.type==='reply'){awaitingResponse=false;manualBtn.disabled=false;manualBtn.textContent='Начать реплику вручную';addMessage('bot',data.text);addFeedback(data.assessment||{})}if(data.type==='state')setStatus(data.message,true);if(data.type==='error'){awaitingResponse=false;manualBtn.disabled=false;manualBtn.textContent='Начать реплику вручную';const now=Date.now();if(data.message!==lastError||now-lastErrorAt>1500){addMessage('system',data.message);lastError=data.message;lastErrorAt=now}setStatus('Слушаю следующую реплику.',true)}};ws.onclose=()=>{if(running)addMessage('system','Соединение с локальным голосовым сервером закрыто.');stop()};processor.onaudioprocess=event=>{if(!running||!ws||ws.readyState!==WebSocket.OPEN)return;const input=event.inputBuffer.getChannelData(0);let sum=0;for(let i=0;i<input.length;i++)sum+=input[i]*input[i];const rms=Math.sqrt(sum/input.length);level.style.width=Math.min(100,rms*850)+'%';if(awaitingResponse)return;const pcm=toPCM16(downsample(input,audioContext.sampleRate));const now=Date.now();if(!speech){if(!botSpeaking&&rms<Math.max(0.03,noiseFloor*3)){noiseFloor=Math.min(0.04,Math.max(0.002,noiseFloor*0.96+rms*0.04))}const onsetThreshold=Math.max(0.006,noiseFloor*1.8)*(botSpeaking?1.5:1);const hot=rms>onsetThreshold;if(hot){hotFrames++;lastVoiceAt=now}else hotFrames=0;preRoll.push(pcm);if(preRoll.length>4)preRoll.shift();if(hotFrames>=2)beginSpeech(false)}else{ws.send(pcm.buffer);const releaseThreshold=Math.max(0.005,noiseFloor*1.2);if(rms>releaseThreshold){lastVoiceAt=now;quietFrames=0}else quietFrames++;if(manualCapture){if(now-speechStartedAt>30000)endSpeech('timeout')}else if(quietFrames>=3&&now-lastVoiceAt>SILENCE_MS)endSpeech('silence');else if(now-speechStartedAt>MAX_UTTERANCE_MS)endSpeech('timeout')}}}catch(error){startBtn.disabled=false;setStatus('Микрофон недоступен');addMessage('system','Не удалось запустить голосовой режим: '+error.message)}}
function stop(){running=false;if(speech)sendJson({type:'interrupt'});speech=false;manualCapture=false;awaitingResponse=false;stopPlayback();if(processor)processor.disconnect();if(source)source.disconnect();if(stream)stream.getTracks().forEach(t=>t.stop());if(ws&&ws.readyState<2){sendJson({type:'finish'});ws.close()}if(audioContext)audioContext.close();startBtn.disabled=false;manualBtn.disabled=true;manualBtn.textContent='Начать реплику вручную';manualBtn.className='secondary';stopBtn.disabled=true;dot.className='dot';setStatus('Диалог завершён')}
startBtn.onclick=start;manualBtn.onclick=()=>speech?endSpeech('manual'):beginSpeech(true);stopBtn.onclick=stop;window.addEventListener('beforeunload',stop);
</script>
</body></html>
"""
    return (
        template.replace("__CONFIG__", safe_config)
        .replace(
            "__TOPIC__",
            escape(str(component_config.get("topic") or "Speaking practice"), quote=True),
        )
    )


def realtime_voice_component_html(config: dict[str, Any]) -> str:
    """Render the browser half of the OpenAI Realtime WebRTC voice mode."""
    component_config = dict(config)
    component_config.setdefault("backend_url", "http://127.0.0.1:8766")
    safe_config = json.dumps(component_config, ensure_ascii=False).replace("<", "\\u003c")
    template = r"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<style>
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Arial,sans-serif;color:#18231f;background:#f5f6f1}
.shell{border:1px solid #dfe5df;border-radius:22px;background:white;overflow:hidden;box-shadow:0 12px 28px rgba(20,55,43,.08)}
.top{padding:20px 22px;background:linear-gradient(120deg,#102d49,#176b50);color:white;display:flex;justify-content:space-between;gap:18px;align-items:center}
.top h2{font-size:20px;margin:0 0 5px}.top p{font-size:13px;margin:0;color:#d9e9e4}.cloud{font-size:10px;font-weight:800;color:#c9f26b;text-align:right;letter-spacing:.3px;white-space:nowrap}
.status{display:flex;align-items:center;gap:9px;padding:12px 22px;border-bottom:1px solid #edf0ed;font-size:13px}.dot{width:10px;height:10px;border-radius:50%;background:#b1b8b4}.dot.live{background:#35c58b;box-shadow:0 0 0 6px rgba(53,197,139,.12)}
.meter{height:5px;flex:1;background:#e7ece8;border-radius:10px;overflow:hidden}.meter i{display:block;height:100%;width:0;background:linear-gradient(90deg,#35c58b,#c9f26b);transition:width .06s}
.chat{height:390px;overflow:auto;padding:18px 20px;background:#fafbf8}.msg{max-width:82%;padding:11px 14px;border-radius:16px;margin:0 0 12px;line-height:1.42;font-size:14px;white-space:pre-wrap}.bot{background:#e8f4ef;border-bottom-left-radius:5px}.bot.streaming{opacity:.75}.student{background:#173f31;color:white;margin-left:auto;border-bottom-right-radius:5px}.system{background:#fff4db;color:#654d13;margin-left:auto;margin-right:auto;font-size:12px}.feedback{margin:-6px 0 14px auto;max-width:82%;font-size:12px;color:#53635c;border-left:3px solid #c9f26b;padding:8px 10px;background:#f5f8ee}.feedback.pending{color:#7a847f;border-left-color:#d8dfda;background:#f7f8f5}
.controls{padding:15px 20px 18px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.controls button{border:0;border-radius:12px;padding:11px 16px;font-weight:700;cursor:pointer}.primary{background:#176b50;color:white}.danger{background:#f5e3df;color:#9c392c}.controls button:disabled{opacity:.45;cursor:not-allowed}.hint{font-size:12px;color:#67766f;flex:1;min-width:260px}.scores{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:0 20px 18px}.score{background:#f3f6f2;border-radius:11px;padding:8px 10px}.score b{display:block;font-size:15px}.score span{font-size:10px;color:#68766f}
</style>
</head>
<body>
<div class="shell">
  <div class="top"><div><h2>Голосовая Viva · __TOPIC__</h2><p>Живой разговор: можно говорить естественно, делать паузы и перебивать собеседника.</p></div><div class="cloud">OPENAI REALTIME<br>WEBRTC · SPEECH-TO-SPEECH</div></div>
  <div class="status"><span id="dot" class="dot"></span><span id="status">Нажмите «Начать живой диалог»</span><div class="meter"><i id="level"></i></div></div>
  <div id="chat" class="chat"><div class="msg system">После подключения собеседник сам начнёт разговор по выбранной теме.</div></div>
  <div id="scores" class="scores" style="display:none"></div>
  <div class="controls"><button id="start" class="primary">Начать живой диалог</button><button id="stop" class="danger" disabled>Завершить</button><div class="hint">Пауза до пяти секунд остаётся внутри одной реплики. Если вы заговорите во время ответа, модель остановится и выслушает вас. Служебные фразы вроде “Are you listening?” не влияют на оценку.</div></div>
</div>
<script>
const CONFIG=__CONFIG__;
let pc,dc,stream,audioContext,analyser,source,animationFrame,remoteAudio,running=false,realtimeCallId='';
let greetingPending=true,speechStartedAt=0,streamingBot=null,streamingBotText='',nextTurnNumber=1,saveChain=Promise.resolve();
const pendingTurns=[],assistantBacklog=[],durationBacklog=[],durationByItemId=new Map(),seenInputItems=new Set();
const statusEl=document.getElementById('status'),dot=document.getElementById('dot'),level=document.getElementById('level'),chat=document.getElementById('chat'),startBtn=document.getElementById('start'),stopBtn=document.getElementById('stop');
function setStatus(text,live=false){statusEl.textContent=text;dot.className='dot'+(live?' live':'')}
function addMessage(kind,text){const node=document.createElement('div');node.className='msg '+kind;node.textContent=text;chat.appendChild(node);chat.scrollTop=chat.scrollHeight;return node}
function bridgeUrl(path){return CONFIG.backend_url+path+'?bridge_token='+encodeURIComponent(CONFIG.bridge_token)}
async function responseError(response){try{const data=await response.json();return data.error||('HTTP '+response.status)}catch(e){return 'HTTP '+response.status}}
function isServiceTurn(text){const normalized=(text||'').toLowerCase().trim().replace(/\s+/g,' '),words=normalized.match(/[a-z]+(?:'[a-z]+)?/g)||[];if(words.length<3)return true;return /^(are|can|could|do|did|will|would)\s+you\b/.test(normalized)||/^(please\s+)?(listen|wait|hold on|let me)\b/.test(normalized)||/^(sorry[, ]+)?(are you there|can you hear me)\b/.test(normalized)||/\b(check|listen to|look at)\s+(my|this|the)\s+(example|answer)\b/.test(normalized)||/\bhow do you\s+(like|find|rate)\b/.test(normalized)||/\bwhy do you\s+(ask|repeat)\b/.test(normalized)||/^(but\s+)?i\s+(already\s+)?(gave|answered|said)\s+you\b/.test(normalized)}
function addAssessment(result,turn){const assessment=result.assessment||{},metrics=result.metrics||{},node=turn.feedbackNode;if(result.skipped||!assessment.scoring_available){if(node)node.remove();return}const quote=assessment.evidence_quote?' Основание: «'+assessment.evidence_quote+'».':'';node.className='feedback';node.textContent=assessment.feedback_ru+(assessment.correction_en?' Исправление: '+assessment.correction_en:'')+quote;chat.scrollTop=chat.scrollHeight;const scores=[['Грамматика',assessment.grammar_score],['Словарь',assessment.vocabulary_score],['По теме',assessment.relevance_score],['Темп',metrics.fluency_score]];const box=document.getElementById('scores');box.innerHTML='';scores.forEach(([name,value])=>{const n=document.createElement('div');n.className='score';n.innerHTML='<b>'+Math.round((value||0)*100)+'%</b><span>'+name+'</span>';box.appendChild(n)});box.style.display='grid'}
async function saveTurn(turn){try{const response=await fetch(bridgeUrl('/turn'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:CONFIG.session_id,student_id:CONFIG.student_id,assignment_id:CONFIG.assignment_id,student_text:turn.text,assistant_text:turn.assistantText,duration_seconds:turn.duration,realtime_call_id:realtimeCallId,client_turn_id:turn.id}),keepalive:true});if(!response.ok)throw new Error(await responseError(response));addAssessment(await response.json(),turn)}catch(error){if(turn.feedbackNode){turn.feedbackNode.className='feedback';turn.feedbackNode.textContent='Оценку этой реплики сохранить не удалось: '+error.message}}}
function queueSave(turn){if(!turn.scorable||turn.saveQueued)return;turn.saveQueued=true;saveChain=saveChain.then(()=>saveTurn(turn))}
function attachAssistant(assistant){const turn=pendingTurns.find(item=>!item.assistantText);if(!turn){assistantBacklog.push(assistant);return}turn.assistantText=assistant.text;turn.assistantNode=assistant.node;if(turn.feedbackNode&&assistant.node)assistant.node.insertAdjacentElement('afterend',turn.feedbackNode);queueSave(turn)}
function registerStudent(text,event){const itemId=event.item_id||'',dedupeKey=itemId||text;if(seenInputItems.has(dedupeKey))return;seenInputItems.add(dedupeKey);const duration=durationByItemId.get(itemId)||durationBacklog.shift()||0;if(itemId)durationByItemId.delete(itemId);const studentNode=addMessage('student',text),scorable=!isServiceTurn(text),turn={id:'voice-turn-'+nextTurnNumber++,itemId:itemId,text:text,duration:duration,studentNode:studentNode,assistantText:'',assistantNode:null,feedbackNode:null,scorable:scorable,saveQueued:false};if(scorable){turn.feedbackNode=document.createElement('div');turn.feedbackNode.className='feedback pending';turn.feedbackNode.textContent='Независимая проверка появится под ответом собеседника.';chat.appendChild(turn.feedbackNode)}pendingTurns.push(turn);if(assistantBacklog.length){const assistant=assistantBacklog.shift();if(assistant.node)chat.insertBefore(studentNode,assistant.node);attachAssistant(assistant)}chat.scrollTop=chat.scrollHeight}
function finishStreamingBot(text){let node;if(streamingBot){streamingBot.textContent=text;streamingBot.className='msg bot';node=streamingBot;streamingBot=null;streamingBotText=''}else node=addMessage('bot',text);return node}
function handleEvent(event){
  if(event.type==='input_audio_buffer.speech_started'){speechStartedAt=performance.now();setStatus('Слышу вас…',true)}
  if(event.type==='input_audio_buffer.speech_stopped'){const timed=(Number(event.audio_end_ms)-Number(event.audio_start_ms))/1000;const duration=Number.isFinite(timed)&&timed>0?timed:(speechStartedAt?Math.max(.1,(performance.now()-speechStartedAt)/1000-5.2):0);if(event.item_id)durationByItemId.set(event.item_id,duration);else durationBacklog.push(duration);speechStartedAt=0;setStatus('Мысль принята — собеседник отвечает…',true)}
  if(event.type==='conversation.item.input_audio_transcription.delta')setStatus('Распознаю речь в реальном времени…',true)
  if(event.type==='conversation.item.input_audio_transcription.completed'){const text=(event.transcript||'').trim();if(text)registerStudent(text,event)}
  if(event.type==='response.created')setStatus('Собеседник формирует ответ…',true)
  if(event.type==='response.output_audio_transcript.delta'){if(!streamingBot){streamingBot=addMessage('bot streaming','');streamingBotText=''}streamingBotText+=event.delta||'';streamingBot.textContent=streamingBotText;chat.scrollTop=chat.scrollHeight}
  if(event.type==='response.output_audio_transcript.done'){const text=(event.transcript||streamingBotText||'').trim(),node=finishStreamingBot(text);if(greetingPending){greetingPending=false}else if(text){attachAssistant({text:text,node:node})}setStatus('Слушаю вас. Говорите естественно.',true)}
  if(event.type==='output_audio_buffer.started')setStatus('Собеседник говорит — его можно перебить.',true)
  if(event.type==='output_audio_buffer.stopped')setStatus('Слушаю вас. Говорите естественно.',true)
  if(event.type==='error'){const message=(event.error&&event.error.message)||'Неизвестная ошибка Realtime API';addMessage('system',message);setStatus('Ошибка соединения')}
}
function updateMeter(){if(!running||!analyser)return;const values=new Uint8Array(analyser.fftSize);analyser.getByteTimeDomainData(values);let sum=0;for(const value of values){const sample=(value-128)/128;sum+=sample*sample}const rms=Math.sqrt(sum/values.length);level.style.width=Math.min(100,rms*900)+'%';animationFrame=requestAnimationFrame(updateMeter)}
async function start(){startBtn.disabled=true;setStatus('Запрашиваю микрофон и создаю защищённый WebRTC-канал…',true);try{
  stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1},video:false});
  audioContext=new (window.AudioContext||window.webkitAudioContext)();await audioContext.resume();source=audioContext.createMediaStreamSource(stream);analyser=audioContext.createAnalyser();analyser.fftSize=512;source.connect(analyser);
  pc=new RTCPeerConnection();remoteAudio=document.createElement('audio');remoteAudio.autoplay=true;remoteAudio.playsInline=true;pc.ontrack=event=>{remoteAudio.srcObject=event.streams[0]};stream.getAudioTracks().forEach(track=>pc.addTrack(track,stream));
  dc=pc.createDataChannel('oai-events');dc.addEventListener('message',message=>{try{handleEvent(JSON.parse(message.data))}catch(error){addMessage('system','Не удалось обработать событие диалога.')}});dc.addEventListener('open',()=>{running=true;stopBtn.disabled=false;setStatus('Подключено. Собеседник начинает разговор…',true);updateMeter();dc.send(JSON.stringify({type:'response.create',response:{instructions:'Start now: greet the student, explicitly name the selected lesson topic, and ask for one original English example. Keep this opening to two short sentences.'}}))});dc.addEventListener('close',()=>{if(running){addMessage('system','Realtime-соединение закрыто.');stop()}});
  const offer=await pc.createOffer();await pc.setLocalDescription(offer);const url=bridgeUrl('/session')+'&student_id='+encodeURIComponent(CONFIG.student_id)+'&assignment_id='+encodeURIComponent(CONFIG.assignment_id)+'&session_id='+encodeURIComponent(CONFIG.session_id);const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/sdp'},body:offer.sdp});if(!response.ok)throw new Error(await responseError(response));realtimeCallId=response.headers.get('X-Realtime-Call-Id')||'';await pc.setRemoteDescription({type:'answer',sdp:await response.text()});
}catch(error){startBtn.disabled=false;setStatus('Не удалось запустить диалог');addMessage('system','Запуск Realtime не удался: '+error.message);cleanup()}}
function cleanup(){running=false;if(animationFrame)cancelAnimationFrame(animationFrame);if(dc&&dc.readyState!=='closed')dc.close();if(pc)pc.close();if(stream)stream.getTracks().forEach(track=>track.stop());if(audioContext)audioContext.close();if(remoteAudio){remoteAudio.pause();remoteAudio.srcObject=null}level.style.width='0';startBtn.disabled=false;stopBtn.disabled=true;dot.className='dot'}
function stop(){if(!running){cleanup();return}fetch(bridgeUrl('/finish'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:CONFIG.session_id}),keepalive:true}).catch(()=>{});cleanup();setStatus('Диалог завершён')}
startBtn.onclick=start;stopBtn.onclick=stop;window.addEventListener('beforeunload',stop);
</script>
</body></html>
"""
    return (
        template.replace("__CONFIG__", safe_config)
        .replace(
            "__TOPIC__",
            escape(str(component_config.get("topic") or "Speaking practice"), quote=True),
        )
    )
