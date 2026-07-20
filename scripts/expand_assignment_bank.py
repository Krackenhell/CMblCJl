from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "english_b2_assignments.json"

COMMON = {
    "eng_present_perfect": (
        "Present Perfect и Past Simple",
        "Времена и связь с настоящим",
        ["Present Perfect связан с настоящим или незавершённым периодом", "Past Simple относится к завершённому времени", "верная форма причастия"],
        ["Present Perfect с законченным временем", "Past Simple с yet или since"],
    ),
    "eng_conditionals": (
        "Условные предложения",
        "Реальные и гипотетические условия",
        ["верная форма в if-clause", "верная форма главной части", "смысл типа условного предложения"],
        ["will после if", "would в обеих частях"],
    ),
    "eng_passive": (
        "Passive Voice",
        "Страдательный залог",
        ["форма be сохраняет исходное время", "использовано past participle", "объект стал подлежащим"],
        ["пропущен be", "изменено исходное время"],
    ),
    "eng_reported_speech": (
        "Reported Speech",
        "Косвенная речь",
        ["корректный backshift", "изменены местоимения и указатели времени", "порядок слов утвердительного предложения"],
        ["сохранён вопросительный порядок", "не изменены указатели времени"],
    ),
    "eng_modals_deduction": (
        "Модальные глаголы предположения",
        "Модальные глаголы предположения",
        ["must для сильного положительного вывода", "can't для сильного отрицательного вывода", "might/could для возможности", "modal + have + V3 для прошлого"],
        ["mustn't вместо can't", "пропущено have в выводе о прошлом"],
    ),
    "eng_relative_clauses": (
        "Relative Clauses",
        "Определительные придаточные",
        ["who для людей", "which для предметов", "where для мест", "запятые в non-defining clause"],
        ["that между запятыми", "пропущены запятые вокруг дополнительной информации"],
    ),
    "eng_gerund_infinitive": (
        "Gerund и Infinitive",
        "Формы после глаголов",
        ["форма выбрана по управляющему глаголу", "учтено изменение смысла", "использована полная форма"],
        ["формы считаются взаимозаменяемыми", "выбор сделан только по глаголу в скобках"],
    ),
    "eng_articles": (
        "Артикли в контексте",
        "A/an, the и нулевой артикль",
        ["a/an для первого неспецифичного упоминания", "the для определённого или повторного объекта", "нулевой артикль для общего понятия"],
        ["the при первом неспецифичном упоминании", "a/an с общим неисчисляемым понятием"],
    ),
}


TASKS = {
    "eng_present_perfect": [
        ("поездка", "Complete: 1) I ___ already ___ (book) the tickets. 2) We ___ (meet) the guide yesterday. 3) She ___ (live) in Rome since 2021. 4) ___ you ___ (see) him last night?", "1) have already booked; 2) met; 3) has lived; 4) Did, see."),
        ("рабочая переписка", "Complete: 1) He ___ (not reply) yet. 2) I ___ (send) the file on Monday. 3) We ___ (know) each other for ten years. 4) The printer ___ (break) last week.", "1) hasn't replied; 2) sent; 3) have known; 4) broke."),
        ("карьера", "Complete: 1) She ___ just ___ (complete) the course. 2) They ___ (launch) the product in 2024. 3) I ___ never ___ (work) abroad. 4) Ben ___ (join) the team two months ago.", "1) has just completed; 2) launched; 3) have never worked; 4) joined."),
        ("опыт", "Complete: 1) Mia ___ (go) out, so she is unavailable now. 2) She ___ (return) an hour ago. 3) ___ you ever ___ (take) a night train? 4) I ___ (not enjoy) my first flight when I was sixteen.", "1) has gone; 2) returned; 3) Have, taken; 4) didn't enjoy."),
        ("результаты", "Complete: 1) We ___ (write) three reports so far. 2) I ___ (write) the summary last month. 3) The parcel ___ (not arrive) yet. 4) The workshop ___ (start) on Tuesday.", "1) have written; 2) wrote; 3) hasn't arrived; 4) started."),
        ("изменения", "Complete: 1) I ___ (work) here since May. 2) The company ___ (change) its policy in 2022. 3) Our results ___ recently ___ (improve). 4) Lara ___ (not attend) yesterday's meeting.", "1) have worked; 2) changed; 3) have recently improved; 4) didn't attend."),
    ],
    "eng_conditionals": [
        ("планы", "Complete: 1) If it ___ (rain), we ___ (stay) home. 2) If I ___ (have) more time, I ___ (learn) Italian. 3) If you heat ice, it ___ (melt). 4) If she ___ (call), tell me.", "1) rains, will stay; 2) had, would learn; 3) melts; 4) calls."),
        ("работа", "Complete: 1) If we ___ (finish) early, we ___ (send) the draft today. 2) If I ___ (be) you, I ___ (ask) for feedback. 3) Unless he ___ (hurry), he ___ (miss) the train. 4) Water boils if it ___ (reach) 100°C.", "1) finish, will send; 2) were, would ask; 3) hurries, will miss; 4) reaches."),
        ("учёба", "Complete: 1) If Nina ___ (revise), she ___ (pass) the exam. 2) If classes ___ (be) smaller, students ___ (get) more feedback. 3) If you mix blue and yellow, you ___ (get) green. 4) What ___ you ___ (do) if you lost your notes?", "1) revises, will pass; 2) were, would get; 3) get; 4) would, do."),
        ("путешествие", "Complete: 1) If the flight ___ (be) delayed, we ___ (take) the train. 2) If I ___ (speak) Japanese, I ___ (travel) alone. 3) If you ___ (not book) now, prices will rise. 4) Plants die if they ___ (not get) water.", "1) is, will take; 2) spoke, would travel; 3) don't book; 4) don't get."),
        ("продукт", "Complete: 1) If users ___ (like) the demo, we ___ (build) the feature. 2) If the budget ___ (be) larger, we ___ (hire) a designer. 3) If you press this button, the app ___ (restart). 4) I would test again if the error ___ (appear).", "1) like, will build; 2) were, would hire; 3) restarts; 4) appeared."),
        ("экология", "Complete: 1) If cities ___ (improve) transport, pollution ___ (fall). 2) If I ___ (own) a car, I still ___ (cycle) to work. 3) Ice melts if the temperature ___ (rise). 4) Unless we ___ (act), the problem ___ (grow).", "1) improve, will fall; 2) owned, would cycle; 3) rises; 4) act, will grow."),
    ],
    "eng_passive": [
        ("музей", "Rewrite in the passive: 1) They built the museum in 1998. 2) Experts are restoring the main hall. 3) Someone has damaged the door. 4) They will reopen it in June.", "1) The museum was built in 1998. 2) The main hall is being restored. 3) The door has been damaged. 4) It will be reopened in June."),
        ("сервис", "Rewrite in the passive: 1) The team updates the database daily. 2) Engineers are fixing the server. 3) They sent the warning yesterday. 4) Someone had deleted the backup.", "1) The database is updated daily. 2) The server is being fixed. 3) The warning was sent yesterday. 4) The backup had been deleted."),
        ("исследование", "Rewrite in the passive: 1) Scientists collected the samples. 2) They have published the results. 3) The lab will repeat the experiment. 4) Reviewers are checking the paper.", "1) The samples were collected. 2) The results have been published. 3) The experiment will be repeated. 4) The paper is being checked."),
        ("мероприятие", "Rewrite in the passive: 1) Volunteers organise the festival. 2) They cancelled two concerts. 3) The organisers have added a workshop. 4) They will stream the final online.", "1) The festival is organised by volunteers. 2) Two concerts were cancelled. 3) A workshop has been added. 4) The final will be streamed online."),
        ("доставка", "Rewrite in the passive: 1) The courier delivered the parcel. 2) Staff are packing the orders. 3) They have changed the address. 4) The company will refund the fee.", "1) The parcel was delivered by the courier. 2) The orders are being packed. 3) The address has been changed. 4) The fee will be refunded by the company."),
        ("образование", "Rewrite in the passive: 1) Tutors assess every project. 2) They recorded the lecture yesterday. 3) The school has introduced a new course. 4) Teachers are preparing the materials.", "1) Every project is assessed by tutors. 2) The lecture was recorded yesterday. 3) A new course has been introduced. 4) The materials are being prepared."),
    ],
    "eng_reported_speech": [
        ("проект", "Report the statements: 1) Ava said, ‘I am testing the app.’ 2) Max said, ‘I finished it yesterday.’ 3) Eva asked, ‘Do you need help?’ 4) Tom said, ‘I will call tomorrow.’", "1) Ava said that she was testing the app. 2) Max said that he had finished it the day before. 3) Eva asked if/whether I needed help. 4) Tom said that he would call the next day."),
        ("поездка", "Report the statements: 1) Leo said, ‘I have lost my ticket.’ 2) Mia said, ‘We are leaving today.’ 3) Sam asked, ‘Where is the platform?’ 4) Nina said, ‘I can meet you here.’", "1) Leo said that he had lost his ticket. 2) Mia said that they were leaving that day. 3) Sam asked where the platform was. 4) Nina said that she could meet me there."),
        ("курс", "Report the statements: 1) Kim said, ‘I don't understand this task.’ 2) Dan said, ‘I watched the lecture last night.’ 3) Amy asked, ‘Have you submitted the essay?’ 4) Rob said, ‘I may join later.’", "1) Kim said that she didn't understand that task. 2) Dan said that he had watched the lecture the previous night. 3) Amy asked if/whether I had submitted the essay. 4) Rob said that he might join later."),
        ("офис", "Report the statements: 1) Ann said, ‘I am working here now.’ 2) Joe said, ‘We bought this printer yesterday.’ 3) Ben asked, ‘When will the meeting start?’ 4) Liz said, ‘I must finish today.’", "1) Ann said that she was working there then. 2) Joe said that they had bought that printer the day before. 3) Ben asked when the meeting would start. 4) Liz said that she had to finish that day."),
        ("интервью", "Report the statements: 1) Sara said, ‘I have worked abroad.’ 2) Mark said, ‘I left my job last month.’ 3) The manager asked, ‘Can you start tomorrow?’ 4) Sara said, ‘I will think about it.’", "1) Sara said that she had worked abroad. 2) Mark said that he had left his job the previous month. 3) The manager asked if/whether I could start the next day. 4) Sara said that she would think about it."),
        ("новости", "Report the statements: 1) Ian said, ‘The storm is getting worse.’ 2) Fay said, ‘They closed the road this morning.’ 3) Lee asked, ‘Is the airport open?’ 4) Ian said, ‘We might stay here tonight.’", "1) Ian said that the storm was getting worse. 2) Fay said that they had closed the road that morning. 3) Lee asked if/whether the airport was open. 4) Ian said that they might stay there that night."),
    ],
    "eng_modals_deduction": [
        ("офис", "Choose must, might/could, can't or a past form: 1) Her laptop is here; she ___ be nearby. 2) The office is locked; they ___ be inside. 3) I am unsure; he ___ know the answer. 4) The file appeared overnight; Maya ___ have uploaded it.", "1) must; 2) can't; 3) might/could; 4) must have."),
        ("дорога", "Choose must, might/could, can't or a past form: 1) The road is wet; it ___ have rained. 2) His car is gone; he ___ be at home. 3) This route ___ be faster, but I am unsure. 4) She arrived in ten minutes; she ___ have walked.", "1) must have; 2) can't; 3) might/could; 4) can't have."),
        ("доставка", "Choose must, might/could, can't or a past form: 1) The parcel is marked delivered; it ___ be downstairs. 2) The address is wrong; this ___ be my order. 3) The courier ___ have called, but I am not certain. 4) The box is empty; someone ___ have opened it.", "1) must; 2) can't; 3) might/could have; 4) must have."),
        ("экзамен", "Choose must, might/could, can't or a past form: 1) She got every answer right; she ___ know the topic well. 2) He was abroad; he ___ have attended in person. 3) I am unsure; the result ___ change. 4) Their names are absent; they ___ have registered.", "1) must; 2) can't have; 3) might/could; 4) can't have."),
        ("дом", "Choose must, might/could, can't or a past form: 1) Music is playing; someone ___ be home. 2) The keys are on the table; they ___ be lost. 3) The noise ___ be the washing machine. 4) The window is broken; a bird ___ have hit it.", "1) must; 2) can't; 3) might/could; 4) might/could have."),
        ("встреча", "Choose must, might/could, can't or a past form: 1) Everyone is seated; the talk ___ be starting. 2) The room is empty; this ___ be the right venue. 3) Alex ___ arrive later; he has not confirmed. 4) The slides are ready; Priya ___ have finished them.", "1) must; 2) can't; 3) might/could; 4) must have."),
    ],
    "eng_relative_clauses": [
        ("профессии и техника", "Combine: 1) The engineer designed the bridge. She won an award. 2) My phone is waterproof. It cost £300. 3) This is the studio. We recorded the podcast there. 4) Leo lives in York. He is my cousin.", "1) The engineer who designed the bridge won an award. 2) My phone, which cost £300, is waterproof. 3) This is the studio where/in which we recorded the podcast. 4) Leo, who lives in York, is my cousin."),
        ("работа", "Combine: 1) Employees work remotely. They need secure access. 2) Our director speaks Japanese. She joined in May. 3) The office has closed. We met there. 4) The tool saves time. It was released yesterday.", "1) Employees who work remotely need secure access. 2) Our director, who joined in May, speaks Japanese. 3) The office where/in which we met has closed. 4) The tool, which was released yesterday, saves time."),
        ("путешествие", "Combine: 1) The guide helped us. He grew up locally. 2) Kyoto has many temples. It was once Japan's capital. 3) That is the hotel. We stayed there. 4) Travellers book early. They get lower prices.", "1) The guide who grew up locally helped us. 2) Kyoto, which was once Japan's capital, has many temples. 3) That is the hotel where/in which we stayed. 4) Travellers who book early get lower prices."),
        ("образование", "Combine: 1) The tutor teaches grammar. She wrote this book. 2) The course is free. It lasts six weeks. 3) The library is being renovated. We usually study there. 4) Students ask questions. They learn faster.", "1) The tutor who wrote this book teaches grammar. 2) The course, which lasts six weeks, is free. 3) The library where/in which we usually study is being renovated. 4) Students who ask questions learn faster."),
        ("технологии", "Combine: 1) The developer fixed the bug. He joined yesterday. 2) My tablet has stopped working. It is only a year old. 3) This is the page. Users reset passwords there. 4) Apps collect less data. They earn more trust.", "1) The developer who joined yesterday fixed the bug. 2) My tablet, which is only a year old, has stopped working. 3) This is the page where/in which users reset passwords. 4) Apps which collect less data earn more trust."),
        ("культура", "Combine: 1) The actor played Hamlet. He visited our school. 2) The film won three awards. It was shot locally. 3) The theatre has reopened. We saw the play there. 4) Artists challenge conventions. They often face criticism.", "1) The actor who played Hamlet visited our school. 2) The film, which was shot locally, won three awards. 3) The theatre where/in which we saw the play has reopened. 4) Artists who challenge conventions often face criticism."),
    ],
    "eng_gerund_infinitive": [
        ("намерения", "Complete: 1) I decided ___ (apply) today. 2) She avoids ___ (drive) at night. 3) Remember ___ (lock) the door. 4) I remember ___ (visit) this place as a child.", "1) to apply; 2) driving; 3) to lock; 4) visiting."),
        ("перерыв", "Complete: 1) Please stop ___ (make) that noise. 2) We stopped ___ (take) a photograph. 3) He promised ___ (help). 4) They suggested ___ (leave) early.", "1) making; 2) to take; 3) to help; 4) leaving."),
        ("эксперимент", "Complete: 1) Try ___ (restart) the device as an experiment. 2) Try ___ (finish) before noon. 3) I regret ___ (say) those words. 4) We regret ___ (inform) you that the flight is cancelled.", "1) restarting; 2) to finish; 3) saying; 4) to inform."),
        ("привычки", "Complete: 1) She enjoys ___ (cook). 2) We hope ___ (travel) soon. 3) He admitted ___ (break) the vase. 4) I managed ___ (solve) the problem.", "1) cooking; 2) to travel; 3) breaking; 4) to solve."),
        ("общение", "Complete: 1) Would you mind ___ (wait)? 2) They agreed ___ (meet) online. 3) I miss ___ (talk) to my old colleagues. 4) She refused ___ (answer).", "1) waiting; 2) to meet; 3) talking; 4) to answer."),
        ("обучение", "Complete: 1) The teacher recommended ___ (read) this article. 2) I learned ___ (use) the tool. 3) Keep ___ (practise) every day. 4) We arranged ___ (have) a tutorial.", "1) reading; 2) to use; 3) practising; 4) to have."),
    ],
    "eng_articles": [
        ("город", "Choose a/an, the or —: 1) We booked ___ apartment near the centre. 2) ___ apartment had a balcony. 3) We walked along ___ River Thames. 4) ___ public transport is convenient there.", "1) an; 2) the; 3) the; 4) —."),
        ("работа", "Choose a/an, the or —: 1) She applied for ___ job yesterday. 2) ___ job requires experience. 3) He gave me ___ useful advice. 4) We had lunch in ___ office kitchen.", "1) a; 2) the; 3) —; 4) the."),
        ("природа", "Choose a/an, the or —: 1) We saw ___ eagle above us. 2) ___ eagle landed nearby. 3) ___ Mount Everest is in Asia. 4) ___ air pollution harms health.", "1) an; 2) the; 3) —; 4) —."),
        ("учёба", "Choose a/an, the or —: 1) I borrowed ___ dictionary. 2) ___ dictionary was very old. 3) She studies ___ economics. 4) We met in ___ university library.", "1) a; 2) the; 3) —; 4) the."),
        ("новости", "Choose a/an, the or —: 1) I read ___ interesting article. 2) ___ article was about climate. 3) ___ information was surprising. 4) It mentioned ___ United States.", "1) an; 2) the; 3) —; 4) the."),
        ("еда", "Choose a/an, the or —: 1) We tried ___ local dish. 2) ___ dish contained rice. 3) ___ rice is common in the region. 4) The chef added ___ unusual spice.", "1) a; 2) the; 3) —; 4) an."),
    ],
}


OPEN_TASKS = {
    "eng_linking": (
        "Связность текста",
        "Связность текста и linking words",
        [
            ("удалённая работа", "Write 100–130 words comparing office and remote work. Use although, however, because and therefore accurately."),
            ("городской транспорт", "Write 100–130 words arguing for better public transport. Use moreover, while, as a result and in conclusion."),
            ("онлайн-обучение", "Write 100–130 words about strengths and limits of online learning. Use on the one hand, on the other hand, because and therefore."),
            ("социальные сети", "Write 100–130 words about social media and attention. Use although, nevertheless, since and as a result."),
            ("здоровые привычки", "Write 100–130 words explaining one realistic healthy habit. Use first of all, however, because and consequently."),
            ("экологичный кампус", "Write 100–130 words proposing a greener university campus. Use in addition, whereas, therefore and to sum up."),
        ],
    ),
    "eng_formal_writing": (
        "Деловое письмо",
        "Формальная письменная коммуникация",
        [
            ("перенос интервью", "Write a 100–130 word formal email asking to reschedule a job interview. Explain the reason, offer two times and request confirmation."),
            ("ошибка в счёте", "Write a 100–130 word formal email reporting an incorrect invoice. Identify the charge, attach evidence and request a corrected invoice."),
            ("запрос преподавателю", "Write a 100–130 word formal email asking a lecturer for clarification about an assignment. State the exact ambiguity and your deadline."),
            ("возврат товара", "Write a 100–130 word formal email requesting a refund for a faulty product. Include the order date, fault and preferred resolution."),
            ("партнёрство", "Write a 100–130 word formal email proposing a student-club partnership. Explain the shared benefit and suggest a short meeting."),
            ("пропущенный дедлайн", "Write a 100–130 word formal email about a missed deadline. Take responsibility, give a concise reason and propose a new submission time."),
        ],
    ),
}


def build() -> list[dict]:
    current = json.loads(PATH.read_text(encoding="utf-8"))
    current = [item for item in current if int(item.get("variant", 1)) <= 3]
    additions: list[dict] = []
    for skill_id, rows in TASKS.items():
        title, topic, criteria, errors = COMMON[skill_id]
        for offset, (suffix, instructions, reference) in enumerate(rows, start=4):
            additions.append(
                {
                    "title": f"{title}: {suffix}",
                    "topic_key": skill_id,
                    "difficulty": ((offset - 1) % 3) + 1,
                    "variant": offset,
                    "subject": "Английский язык · B2",
                    "topic": topic,
                    "instructions": instructions.strip(),
                    "starter_code": "",
                    "skill_ids": [skill_id],
                    "rubric": {
                        "reference_answer": reference,
                        "criteria": criteria,
                        "common_errors": errors,
                    },
                }
            )
    for skill_id, (title, topic, rows) in OPEN_TASKS.items():
        for offset, (suffix, instructions) in enumerate(rows, start=4):
            additions.append(
                {
                    "title": f"{title}: {suffix}",
                    "topic_key": skill_id,
                    "difficulty": ((offset - 1) % 3) + 1,
                    "variant": offset,
                    "subject": "Английский язык · B2",
                    "topic": topic,
                    "instructions": instructions,
                    "starter_code": "",
                    "skill_ids": [skill_id],
                    "rubric": {
                        "reference_answer": "A complete B2-level response that fulfils every instruction in the prompt.",
                        "criteria": ["выполнены все пункты задания", "логичная структура и точные связки", "лексика и грамматика уровня B2", "подходящий регистр"],
                        "common_errors": ["пропущен обязательный пункт", "связки перечислены без логической функции", "неподходящий регистр"],
                    },
                }
            )
    result = current + additions
    if len(result) != 90:
        raise RuntimeError(f"Expected 90 assignments, got {len(result)}")
    return result


if __name__ == "__main__":
    PATH.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("English B2 assignment bank expanded to 90 items.")
