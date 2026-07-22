from __future__ import annotations

import json
import os
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
.chat{height:390px;overflow:auto;padding:18px 20px;background:#fafbf8}.msg{max-width:82%;padding:11px 14px;border-radius:16px;margin:0 0 12px;line-height:1.42;font-size:14px;white-space:pre-wrap}.bot{background:#e8f4ef;border-bottom-left-radius:5px}.student{background:#173f31;color:white;margin-left:auto;border-bottom-right-radius:5px}.system{background:#fff4db;color:#654d13;margin-left:auto;margin-right:auto;font-size:12px}.feedback{margin:-6px 0 14px auto;max-width:82%;font-size:12px;color:#53635c;border-left:3px solid #c9f26b;padding:7px 10px;background:#f5f8ee}.controls{padding:15px 20px 18px;display:flex;gap:10px;align-items:center}.controls button{border:0;border-radius:12px;padding:11px 16px;font-weight:700;cursor:pointer}.primary{background:#176b50;color:white}.danger{background:#f5e3df;color:#9c392c}.controls button:disabled{opacity:.45;cursor:not-allowed}.hint{font-size:12px;color:#67766f;flex:1}.scores{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:0 20px 18px}.score{background:#f3f6f2;border-radius:11px;padding:8px 10px}.score b{display:block;font-size:15px}.score span{font-size:10px;color:#68766f}
</style>
</head>
<body>
<div class="shell">
  <div class="top"><div><h2>Голосовая Viva · __TOPIC__</h2><p>Говорите по-английски. Бот слышит вас даже во время своей реплики.</p></div><div class="local">ПОЛНОСТЬЮ ЛОКАЛЬНО<br>Whisper · Qwen · SAPI</div></div>
  <div class="status"><span id="dot" class="dot"></span><span id="status">Нажмите «Начать диалог»</span><div class="meter"><i id="level"></i></div></div>
  <div id="chat" class="chat"><div class="msg bot">Hi! Let’s have a short B2 speaking practice. Explain one idea from today’s topic and give an English example.</div></div>
  <div id="scores" class="scores" style="display:none"></div>
  <div class="controls"><button id="start" class="primary">Начать диалог</button><button id="stop" class="danger" disabled>Завершить</button><div class="hint">Можно перебить бота — воспроизведение остановится, а микрофон продолжит слушать.</div></div>
</div>
<script>
const CONFIG=__CONFIG__;
const SAMPLE_RATE=16000,SILENCE_MS=700,MAX_UTTERANCE_MS=20000;
let ws,audioContext,stream,source,processor,zeroGain,speech=false,speechStartedAt=0,lastVoiceAt=0,hotFrames=0,preRoll=[],currentAudio=null,running=false,botSpeaking=false;
const statusEl=document.getElementById('status'),dot=document.getElementById('dot'),level=document.getElementById('level'),chat=document.getElementById('chat'),startBtn=document.getElementById('start'),stopBtn=document.getElementById('stop');
function setStatus(text,live=false){statusEl.textContent=text;dot.className='dot'+(live?' live':'')}
function addMessage(kind,text){const node=document.createElement('div');node.className='msg '+kind;node.textContent=text;chat.appendChild(node);chat.scrollTop=chat.scrollHeight;return node}
function addFeedback(data){const node=document.createElement('div');node.className='feedback';const audit=(data.grammar_findings||[]).length?' Проверено: независимое правило + локальная Qwen.':' Проверено: локальная Qwen.';node.textContent=data.feedback_ru+(data.correction_en?' Исправление: '+data.correction_en:'')+audit;chat.appendChild(node);chat.scrollTop=chat.scrollHeight;const metrics=data.metrics||{};const scores=[['Грамматика',data.grammar_score],['Словарь',data.vocabulary_score],['Содержание',data.relevance_score],['Беглость',metrics.fluency_score]];const box=document.getElementById('scores');box.innerHTML='';scores.forEach(([name,value])=>{const n=document.createElement('div');n.className='score';n.innerHTML='<b>'+Math.round((value||0)*100)+'%</b><span>'+name+'</span>';box.appendChild(n)});box.style.display='grid'}
function stopPlayback(){if(currentAudio){try{currentAudio.stop()}catch(e){}currentAudio=null}botSpeaking=false}
function downsample(input,inputRate){if(inputRate===SAMPLE_RATE)return input;const ratio=inputRate/SAMPLE_RATE;const length=Math.round(input.length/ratio);const output=new Float32Array(length);let pos=0;for(let i=0;i<length;i++){const next=Math.round((i+1)*ratio);let sum=0,count=0;for(;pos<next&&pos<input.length;pos++){sum+=input[pos];count++}output[i]=count?sum/count:0}return output}
function toPCM16(float32){const out=new Int16Array(float32.length);for(let i=0;i<float32.length;i++){const s=Math.max(-1,Math.min(1,float32[i]));out[i]=s<0?s*32768:s*32767}return out}
function sendJson(data){if(ws&&ws.readyState===WebSocket.OPEN)ws.send(JSON.stringify(data))}
function beginSpeech(){speech=true;speechStartedAt=Date.now();lastVoiceAt=Date.now();if(botSpeaking){stopPlayback();sendJson({type:'interrupt'})}sendJson({type:'speech_start'});preRoll.forEach(chunk=>ws.send(chunk.buffer));preRoll=[];setStatus('Слушаю вашу реплику…',true)}
function endSpeech(){if(!speech)return;speech=false;hotFrames=0;sendJson({type:'speech_end'});setStatus('Распознаю и формирую ответ…',true)}
async function playWave(buffer){if(!running)return;stopPlayback();try{const decoded=await audioContext.decodeAudioData(buffer.slice(0));const node=audioContext.createBufferSource();node.buffer=decoded;node.connect(audioContext.destination);node.onended=()=>{if(currentAudio===node){currentAudio=null;botSpeaking=false;setStatus('Слушаю. Начните говорить.',true)}};currentAudio=node;botSpeaking=true;node.start()}catch(e){addMessage('system','Не удалось воспроизвести локальный голос, текст ответа сохранён.')  }}
async function start(){startBtn.disabled=true;setStatus('Запрашиваю доступ к микрофону…',true);try{stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});audioContext=new (window.AudioContext||window.webkitAudioContext)();await audioContext.resume();source=audioContext.createMediaStreamSource(stream);processor=audioContext.createScriptProcessor(4096,1,1);zeroGain=audioContext.createGain();zeroGain.gain.value=0;source.connect(processor);processor.connect(zeroGain);zeroGain.connect(audioContext.destination);ws=new WebSocket(CONFIG.websocket_url);ws.binaryType='arraybuffer';ws.onopen=()=>{running=true;stopBtn.disabled=false;sendJson({type:'configure',...CONFIG});setStatus('Слушаю. Начните говорить.',true)};ws.onmessage=async event=>{if(typeof event.data!=='string'){await playWave(event.data);return}const data=JSON.parse(event.data);if(data.type==='transcript')addMessage('student',data.text);if(data.type==='reply'){addMessage('bot',data.text);addFeedback(data.assessment||{})}if(data.type==='state')setStatus(data.message,true);if(data.type==='error')addMessage('system',data.message)};ws.onclose=()=>{if(running)addMessage('system','Соединение с локальным голосовым сервером закрыто.');stop()};processor.onaudioprocess=event=>{if(!running||!ws||ws.readyState!==WebSocket.OPEN)return;const input=event.inputBuffer.getChannelData(0);let sum=0;for(let i=0;i<input.length;i++)sum+=input[i]*input[i];const rms=Math.sqrt(sum/input.length);level.style.width=Math.min(100,rms*850)+'%';const pcm=toPCM16(downsample(input,audioContext.sampleRate));const hot=rms>0.018*(botSpeaking?1.7:1);if(hot){hotFrames++;lastVoiceAt=Date.now()}else hotFrames=0;if(!speech){preRoll.push(pcm);if(preRoll.length>3)preRoll.shift();if(hotFrames>=2)beginSpeech()}else{ws.send(pcm.buffer);if(Date.now()-lastVoiceAt>SILENCE_MS||Date.now()-speechStartedAt>MAX_UTTERANCE_MS)endSpeech()}}}catch(error){startBtn.disabled=false;setStatus('Микрофон недоступен');addMessage('system','Не удалось запустить голосовой режим: '+error.message)}}
function stop(){running=false;endSpeech();stopPlayback();if(processor)processor.disconnect();if(source)source.disconnect();if(stream)stream.getTracks().forEach(t=>t.stop());if(ws&&ws.readyState<2)ws.close();if(audioContext)audioContext.close();startBtn.disabled=false;stopBtn.disabled=true;dot.className='dot';setStatus('Диалог завершён')}
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
