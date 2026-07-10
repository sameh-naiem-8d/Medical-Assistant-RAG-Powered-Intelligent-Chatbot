from __future__ import annotations


ARABIC_SYMPTOM_SYNONYMS: dict[str, dict[str, list[str]]] = {
    "cough": {
        "formal": ["سعال", "كحة"],
        "egyptian": ["بكح", "عندي كحة", "كحة ناشفة", "كحة ببلغم"],
        "misspellings": ["كحه", "كحا", "كحة جامدة", "سعال شديد"],
    },
    "phlegm": {
        "formal": ["بلغم", "مخاط صدري"],
        "egyptian": ["عندي بلغم", "بطلع بلغم", "البلغم كتير"],
        "misspellings": ["بلغمم", "بلغام", "بلغم اخضر", "بلغم أصفر"],
    },
    "mucoid_sputum": {
        "formal": ["بلغم مخاطي"],
        "egyptian": ["بلغم لزج", "بلغم ابيض", "بلغم شفاف"],
        "misspellings": ["بلغم لازق"],
    },
    "rusty_sputum": {
        "formal": ["بلغم بلون الصدأ"],
        "egyptian": ["بلغم بني", "بلغم غامق"],
        "misspellings": ["بلغم مائل للبني", "بلغم احمر غامق"],
    },
    "blood_in_sputum": {
        "formal": ["دم في البلغم", "نفث دم"],
        "egyptian": ["بطلع دم مع الكحة", "بلغم بدم", "كحة بدم"],
        "misspellings": ["دم مع السعال"],
    },
    "breathlessness": {
        "formal": ["ضيق تنفس", "صعوبة في التنفس", "صعوبة التنفس"],
        "egyptian": ["مش قادر اتنفس", "نفسي مقطوع", "مخنوق", "بنهج", "حاسس باختناق"],
        "misspellings": ["ضيق نفس", "صعوبه تنفس", "مش عارف اتنفس", "نهجان", "اختناق"],
    },
    "chest_pain": {
        "formal": ["ألم في الصدر", "الم في الصدر"],
        "egyptian": ["وجع في صدري", "صدري واجعني", "حاسس بضغط على صدري", "نغزة صدر"],
        "misspellings": ["الم صدر", "وجع صدر", "الم بالصدر", "ضغط على الصدر"],
    },
    "throat_irritation": {
        "formal": ["تهيج الحلق", "التهاب الحلق", "ألم الحلق"],
        "egyptian": ["وجع حلق", "حرقان في الزور", "زوري واجعني", "التهاب زور"],
        "misspellings": ["حرقان زور", "وجع زور", "التهاب حلق", "حلقى واجعنى"],
    },
    "patches_in_throat": {
        "formal": ["بقع في الحلق", "صديد على اللوز", "بقع بيضاء في الحلق"],
        "egyptian": ["نقط بيضا في الزور", "صديد على اللوز"],
        "misspellings": ["صديد اللوز", "بقع بيضا"],
    },
    "runny_nose": {
        "formal": ["سيلان الأنف", "سيلان الانف"],
        "egyptian": ["رشح", "مناخيري سايلة", "عندي برد ورشح", "رشحان"],
        "misspellings": ["رشح شديد", "سيلان من الانف"],
    },
    "congestion": {
        "formal": ["احتقان الأنف", "انسداد الأنف"],
        "egyptian": ["مناخيري مسدودة", "زكام", "انسداد في المناخير"],
        "misspellings": ["احتقان", "انسداد الانف", "احتئان"],
    },
    "continuous_sneezing": {
        "formal": ["عطس مستمر"],
        "egyptian": ["بعطس كتير", "عطس ورا بعض", "عندي عطس"],
        "misspellings": ["عطس", "عتس", "عطسان"],
    },
    "sinus_pressure": {
        "formal": ["ضغط في الجيوب الأنفية", "ألم الجيوب الأنفية"],
        "egyptian": ["ضغط على وشي", "وجع حوالين الانف", "صداع الجيوب"],
        "misspellings": ["جيوب انفيه", "جيوب انفية"],
    },
    "loss_of_smell": {
        "formal": ["فقدان الشم", "ضعف حاسة الشم"],
        "egyptian": ["مش بشم", "الشم راح", "مش شامم"],
        "misspellings": ["فقدان شم"],
    },
    "redness_of_eyes": {
        "formal": ["احمرار العين", "احمرار العينين"],
        "egyptian": ["عيني حمرا", "عينيا محمرة"],
        "misspellings": ["احمرار عين"],
    },
    "watering_from_eyes": {
        "formal": ["دموع العين", "تدميع العين"],
        "egyptian": ["عيني بتدمع", "دموع كتير"],
        "misspellings": ["دمع", "دموع العين"],
    },
    "high_fever": {
        "formal": ["حمى شديدة", "حرارة مرتفعة", "ارتفاع شديد في الحرارة"],
        "egyptian": ["سخونية عالية", "حرارتي عالية", "مولع من السخونية", "حرارة عالية"],
        "misspellings": ["سخونيه عاليه", "حراره عاليه", "حمى جامدة", "سخونية جامدة"],
    },
    "mild_fever": {
        "formal": ["حمى خفيفة", "حرارة بسيطة"],
        "egyptian": ["سخونية بسيطة", "حرارة خفيفة"],
        "misspellings": ["سخونيه بسيطة", "حراره بسيطه"],
    },
    "chills": {
        "formal": ["قشعريرة"],
        "egyptian": ["جسمي بيترعش من البرد", "بقشعر", "بردان", "رعشة برد"],
        "misspellings": ["قشعريره", "قشعريرة جامدة"],
    },
    "shivering": {
        "formal": ["ارتعاش", "رجفة"],
        "egyptian": ["برتعش", "جسمي بيرتعش", "رعشة"],
        "misspellings": ["رعشه", "رجفه"],
    },
    "fatigue": {
        "formal": ["تعب", "إرهاق", "إجهاد", "ضعف عام"],
        "egyptian": ["جسمي مكسر", "مش قادر اقوم", "همدان", "تعبان", "حاسس بهبوط", "تكسير في الجسم"],
        "misspellings": ["ارهاق", "اجهاد", "تكسير الجسم", "هبوط", "جسمى مكسر"],
    },
    "malaise": {
        "formal": ["توعك عام", "إعياء عام"],
        "egyptian": ["حاسس اني مش طبيعي", "جسمي مش مظبوط", "تعب عام"],
        "misspellings": ["عدم ارتياح", "اعياء"],
    },
    "lethargy": {
        "formal": ["خمول"],
        "egyptian": ["خامل", "عايز انام طول الوقت", "مش قادر اتحرك"],
        "misspellings": ["خمول شديد"],
    },
    "restlessness": {
        "formal": ["عدم ارتياح", "تململ"],
        "egyptian": ["قلقان ومش عارف اقعد", "متوتر", "مش مستريح"],
        "misspellings": ["تململ"],
    },
    "sweating": {
        "formal": ["تعرق", "عرق شديد"],
        "egyptian": ["بعرق كتير", "عرق بارد", "عرقان"],
        "misspellings": ["تعرئ", "تعرق شديد"],
    },
    "dehydration": {
        "formal": ["جفاف", "جفاف شديد"],
        "egyptian": ["ناشف", "ريقي ناشف", "بقي ناشف", "مش بتبول كتير"],
        "misspellings": ["نشفان", "جفافان"],
    },
    "sunken_eyes": {
        "formal": ["غؤور العين", "عيون غائرة"],
        "egyptian": ["عيني داخلة لجوه", "عينيا غايرة"],
        "misspellings": ["عيون غايرة"],
    },
    "muscle_pain": {
        "formal": ["ألم عضلات", "الم عضلات"],
        "egyptian": ["وجع عضلات", "جسمي واجعني", "عضلاتي واجعاني"],
        "misspellings": ["الام عضلات", "وجع العضلات"],
    },
    "joint_pain": {
        "formal": ["ألم مفاصل", "الم مفاصل"],
        "egyptian": ["وجع مفاصل", "مفاصلي واجعاني", "ركبتي ومفاصلي واجعاني"],
        "misspellings": ["الام مفاصل"],
    },
    "muscle_weakness": {
        "formal": ["ضعف العضلات", "ضعف عضلي"],
        "egyptian": ["عضلاتي ضعيفة", "مش قادر اشيل حاجة"],
        "misspellings": ["ضعف عضلات"],
    },
    "muscle_wasting": {
        "formal": ["ضمور عضلات", "ضمور عضلي"],
        "egyptian": ["العضلات بتصغر", "خسارة في العضلات"],
        "misspellings": ["ضمور في العضلات"],
    },
    "weakness_in_limbs": {
        "formal": ["ضعف الأطراف", "ضعف الاطراف"],
        "egyptian": ["ايدي ورجلي ضعاف", "مش قادر احرك اطرافي"],
        "misspellings": ["ضعف اطراف"],
    },
    "nausea": {
        "formal": ["غثيان"],
        "egyptian": ["نفسي مقلوبة", "حاسس اني هرجع", "قرفان"],
        "misspellings": ["غسيان", "غثيان شديد"],
    },
    "vomiting": {
        "formal": ["قيء", "تقيؤ"],
        "egyptian": ["ترجيع", "بستفرغ", "برجع"],
        "misspellings": ["قئ", "قي", "استفراغ", "ترجيع شديد"],
    },
    "diarrhoea": {
        "formal": ["إسهال", "اسهال"],
        "egyptian": ["بطني سايبة", "بدخل الحمام كتير", "اسهال شديد"],
        "misspellings": ["اسهال", "إسهال شديد"],
    },
    "constipation": {
        "formal": ["إمساك", "امساك"],
        "egyptian": ["مش عارف اعمل حمام", "بطني ممسكة"],
        "misspellings": ["امساك شديد"],
    },
    "abdominal_pain": {
        "formal": ["ألم بالبطن", "ألم في البطن"],
        "egyptian": ["وجع بطن", "بطني واجعاني", "مغص", "بطني بتوجعني"],
        "misspellings": ["الم بطن", "وجع بطني", "مغص جامد"],
    },
    "belly_pain": {
        "formal": ["ألم أسفل البطن", "ألم حول السرة"],
        "egyptian": ["مغص جامد تحت", "وجع حوالين السرة"],
        "misspellings": ["الم اسفل البطن", "وجع اسفل البطن"],
    },
    "stomach_pain": {
        "formal": ["ألم المعدة", "الم المعدة"],
        "egyptian": ["وجع معدة", "معدتي واجعاني"],
        "misspellings": ["الم معده", "وجع معده"],
    },
    "acidity": {
        "formal": ["حموضة", "ارتجاع حمضي"],
        "egyptian": ["بطني محروقة", "حرقان معدة", "حموضة بعد الاكل"],
        "misspellings": ["حموضه", "حرقان معده"],
    },
    "indigestion": {
        "formal": ["عسر هضم"],
        "egyptian": ["الأكل واقف على معدتي", "تخمة", "هضم وحش"],
        "misspellings": ["تخمه"],
    },
    "passage_of_gases": {
        "formal": ["غازات"],
        "egyptian": ["انتفاخ وغازات", "بطني منفوخة"],
        "misspellings": ["غازات", "انتفاخ"],
    },
    "distention_of_abdomen": {
        "formal": ["انتفاخ البطن"],
        "egyptian": ["بطني منفخة", "بطني مشدودة"],
        "misspellings": ["انتفاخ شديد"],
    },
    "swelling_of_stomach": {
        "formal": ["تورم البطن"],
        "egyptian": ["بطني وارمة", "انتفاخ جامد"],
        "misspellings": ["تورم بطن"],
    },
    "pain_during_bowel_movements": {
        "formal": ["ألم أثناء التبرز", "الم اثناء التبرز"],
        "egyptian": ["وجع وانا بعمل حمام"],
        "misspellings": ["ألم مع البراز", "الم مع البراز"],
    },
    "pain_in_anal_region": {
        "formal": ["ألم في منطقة الشرج", "الم في منطقة الشرج"],
        "egyptian": ["وجع عند فتحة الشرج"],
        "misspellings": ["الم شرج", "وجع شرج"],
    },
    "bloody_stool": {
        "formal": ["دم في البراز", "نزيف مع البراز"],
        "egyptian": ["في دم مع البراز", "براز بدم", "دم مع الحمام"],
        "misspellings": ["دم في البراز"],
    },
    "irritation_in_anus": {
        "formal": ["تهيج الشرج", "حكة شرجية"],
        "egyptian": ["حرقان وهرش عند فتحة الشرج"],
        "misspellings": ["حكه شرجيه"],
    },
    "stomach_bleeding": {
        "formal": ["نزيف بالمعدة", "قيء دم"],
        "egyptian": ["ترجيع دم", "براز اسود"],
        "misspellings": ["قيء دم", "قئ دم"],
    },
    "dark_urine": {
        "formal": ["بول داكن"],
        "egyptian": ["البول غامق", "لون البول غامق", "بول غامق"],
        "misspellings": ["بول بني", "بول غامء"],
    },
    "yellow_urine": {
        "formal": ["بول أصفر"],
        "egyptian": ["البول اصفر قوي"],
        "misspellings": ["بول اصفر"],
    },
    "yellowish_skin": {
        "formal": ["اصفرار الجلد", "يرقان الجلد"],
        "egyptian": ["جلدي مصفر", "وشي اصفر", "جلدي اصفر"],
        "misspellings": ["اصفرار جلد"],
    },
    "yellowing_of_eyes": {
        "formal": ["اصفرار العين", "اصفرار العينين"],
        "egyptian": ["عيني صفرا", "بياض العين اصفر"],
        "misspellings": ["صفار العين", "اصفرار عين"],
    },
    "headache": {
        "formal": ["صداع", "ألم الرأس"],
        "egyptian": ["وجع راس", "دماغي وجعاني", "راسي هتنفجر"],
        "misspellings": ["وجع رأس", "صداع جامد"],
    },
    "dizziness": {
        "formal": ["دوخة", "دوار"],
        "egyptian": ["بحس بدوخة", "الدنيا بتلف", "دايخ"],
        "misspellings": ["دوخه", "لفان راس"],
    },
    "spinning_movements": {
        "formal": ["إحساس بالدوران", "احساس بالدوران"],
        "egyptian": ["الدنيا بتلف بيا", "كل حاجة بتلف", "لفان"],
        "misspellings": ["دوار", "لفه"],
    },
    "loss_of_balance": {
        "formal": ["فقدان التوازن", "عدم التوازن"],
        "egyptian": ["مش متزن", "بتمايل", "مش عارف امشي مستقيم"],
        "misspellings": ["عدم اتزان", "عدم توازن"],
    },
    "unsteadiness": {
        "formal": ["عدم ثبات"],
        "egyptian": ["مش ثابت على رجلي", "رجلي مش شايلاني"],
        "misspellings": ["عدم الثبات"],
    },
    "pain_behind_the_eyes": {
        "formal": ["ألم خلف العين", "الم خلف العين"],
        "egyptian": ["وجع ورا العين", "وجع خلف العين"],
        "misspellings": ["الم خلف عيني", "وجع ورا عيني"],
    },
    "blurred_and_distorted_vision": {
        "formal": ["زغللة", "تشوش الرؤية"],
        "egyptian": ["عيني مزغللة", "الرؤية مش واضحة", "النظر مش واضح"],
        "misspellings": ["زغلله", "تشويش رؤية"],
    },
    "visual_disturbances": {
        "formal": ["اضطرابات بصرية", "اضطراب الرؤية"],
        "egyptian": ["بشوف نقط", "بشوف ومضات", "النظر بيتلخبط"],
        "misspellings": ["اضطراب نظر"],
    },
    "lack_of_concentration": {
        "formal": ["ضعف التركيز", "قلة التركيز"],
        "egyptian": ["مش مركز", "ذهني مشتت"],
        "misspellings": ["قلة تركيز"],
    },
    "slurred_speech": {
        "formal": ["تلعثم الكلام", "ثقل الكلام"],
        "egyptian": ["كلامي تقيل", "مش عارف اتكلم كويس", "كلامي متلخبط"],
        "misspellings": ["ثقل كلام", "تقل الكلام"],
    },
    "weakness_of_one_body_side": {
        "formal": ["ضعف في جانب واحد من الجسم", "تنميل في نصف الجسم"],
        "egyptian": ["تنميل في نص جسمي", "ايدي ورجلي ناحية واحدة ضعاف", "نص جسمي متنمل"],
        "misspellings": ["تنميل ناحية واحدة", "شلل نصفي", "تنميل في ايدي", "تنميل في ايدي ورجلي"],
    },
    "altered_sensorium": {
        "formal": ["تغير الوعي", "اضطراب الوعي"],
        "egyptian": ["مش واعي", "ملخبط", "مش مدرك"],
        "misspellings": ["تشتت وعي"],
    },
    "coma": {
        "formal": ["غيبوبة", "فقدان الوعي"],
        "egyptian": ["مش بيفوق", "اغماء طويل"],
        "misspellings": ["غيبوبه", "فقد وعي"],
    },
    "stiff_neck": {
        "formal": ["تيبس الرقبة"],
        "egyptian": ["رقبتي ناشفة", "مش قادر احرك رقبتي"],
        "misspellings": ["تيبس رقبه"],
    },
    "neck_pain": {
        "formal": ["ألم الرقبة", "الم الرقبة"],
        "egyptian": ["وجع رقبة", "رقبتي واجعاني"],
        "misspellings": ["الم رقبه", "وجع رقبه"],
    },
    "back_pain": {
        "formal": ["ألم الظهر", "الم الظهر"],
        "egyptian": ["وجع ظهر", "ضهري واجعني"],
        "misspellings": ["الم ظهر", "وجع ضهر"],
    },
    "palpitations": {
        "formal": ["خفقان القلب"],
        "egyptian": ["قلبي بيدق بسرعة", "دقات قلبي سريعة", "خفقان"],
        "misspellings": ["ضربات قلب سريعة"],
    },
    "fast_heart_rate": {
        "formal": ["سرعة ضربات القلب", "تسارع ضربات القلب"],
        "egyptian": ["قلبي سريع", "نبضي سريع"],
        "misspellings": ["سرعة نبض"],
    },
    "swollen_legs": {
        "formal": ["تورم الساقين", "تورم الرجلين"],
        "egyptian": ["رجلي وارمة", "تورم في الرجلين", "رجليا وارمين"],
        "misspellings": ["تورم رجلين"],
    },
    "swollen_blood_vessels": {
        "formal": ["تورم الأوعية الدموية", "تورم الاوعية الدموية"],
        "egyptian": ["عروق بارزة", "عروق وارمة", "عروق منتفخة"],
        "misspellings": ["عروق طالعة"],
    },
    "prominent_veins_on_calf": {
        "formal": ["بروز أوردة الساق", "دوالي الساق"],
        "egyptian": ["دوالي في الساق", "عروق طالعة في السمانة", "دوالي"],
        "misspellings": ["دوالى", "عروق الساق"],
    },
    "painful_walking": {
        "formal": ["ألم أثناء المشي", "الم اثناء المشي"],
        "egyptian": ["المشي بيوجعني", "وجع مع المشي"],
        "misspellings": ["ألم مع المشي", "الم مع المشي"],
    },
    "cold_hands_and_feets": {
        "formal": ["برودة اليدين والقدمين"],
        "egyptian": ["ايدي ورجلي ساقعين", "اطرافي ساقعة"],
        "misspellings": ["برودة اطراف"],
    },
    "burning_micturition": {
        "formal": ["حرقان أثناء التبول", "حرقان البول"],
        "egyptian": ["حرقان بول", "البول بيحرق"],
        "misspellings": ["حرقه بول", "حرقان اثناء التبول"],
    },
    "bladder_discomfort": {
        "formal": ["ألم المثانة", "الم المثانة"],
        "egyptian": ["وجع مثانة", "ضغط تحت البطن"],
        "misspellings": ["الم مثانه"],
    },
    "continuous_feel_of_urine": {
        "formal": ["إحساس مستمر بالحاجة للتبول"],
        "egyptian": ["حاسس عايز اتبول طول الوقت", "عايز ادخل الحمام طول الوقت"],
        "misspellings": ["احساس بالتبول"],
    },
    "spotting_ urination": {
        "formal": ["نقط دم في البول", "دم في البول"],
        "egyptian": ["نقط دم مع البول", "تبقيع مع البول"],
        "misspellings": ["دم بالبول"],
    },
    "foul_smell_of urine": {
        "formal": ["رائحة كريهة للبول"],
        "egyptian": ["ريحة البول وحشة"],
        "misspellings": ["رائحة بول كريهة"],
    },
    "polyuria": {
        "formal": ["كثرة التبول"],
        "egyptian": ["بتبول كتير", "بدخل الحمام كتير", "تبول كتير"],
        "misspellings": ["كثره تبول", "كثرة تبول"],
    },
    "abnormal_menstruation": {
        "formal": ["اضطراب الدورة الشهرية"],
        "egyptian": ["الدورة مش منتظمة", "نزيف غير طبيعي"],
        "misspellings": ["لخبطة الدورة"],
    },
    "extra_marital_contacts": {
        "formal": ["علاقة جنسية عالية الخطورة"],
        "egyptian": ["علاقة غير آمنة", "علاقة بدون وقاية"],
        "misspellings": ["علاقه غير امنه"],
    },
    "excessive_hunger": {
        "formal": ["جوع شديد"],
        "egyptian": ["جعان جدا", "جوع غير طبيعي", "جوعي شديد"],
        "misspellings": ["جوع شديد"],
    },
    "increased_appetite": {
        "formal": ["زيادة الشهية"],
        "egyptian": ["نفسي مفتوحة", "باكل كتير"],
        "misspellings": ["زيادة شهيه"],
    },
    "loss_of_appetite": {
        "formal": ["فقدان الشهية"],
        "egyptian": ["نفسي مسدودة", "مش عايز اكل", "مليش نفس للاكل"],
        "misspellings": ["فقدان شهيه", "نفسى مسدوده"],
    },
    "irregular_sugar_level": {
        "formal": ["اضطراب مستوى السكر"],
        "egyptian": ["السكر بيعلى وينزل", "سكر مش منتظم", "لخبطة السكر"],
        "misspellings": ["لخبطه السكر"],
    },
    "weight_loss": {
        "formal": ["نقص الوزن", "فقدان الوزن"],
        "egyptian": ["بخس", "وزني بيقل", "خسيت"],
        "misspellings": ["نقصان وزن", "نقص وزن"],
    },
    "weight_gain": {
        "formal": ["زيادة الوزن"],
        "egyptian": ["وزني زاد", "تخنت"],
        "misspellings": ["زيادة وزن"],
    },
    "obesity": {
        "formal": ["سمنة"],
        "egyptian": ["تخن", "وزن زائد"],
        "misspellings": ["سمنه"],
    },
    "enlarged_thyroid": {
        "formal": ["تضخم الغدة الدرقية"],
        "egyptian": ["ورم في الرقبة من الغدة"],
        "misspellings": ["تضخم الغده"],
    },
    "puffy_face_and_eyes": {
        "formal": ["انتفاخ الوجه والعينين"],
        "egyptian": ["وشي منفخ", "عيني منفخة"],
        "misspellings": ["انتفاخ عين"],
    },
    "brittle_nails": {
        "formal": ["هشاشة الأظافر"],
        "egyptian": ["ضوافري بتتكسر", "اظافري بتتكسر"],
        "misspellings": ["اظافر ضعيفة"],
    },
    "swollen_extremeties": {
        "formal": ["تورم الأطراف"],
        "egyptian": ["ايدي ورجلي وارمين", "اطرافي وارمة"],
        "misspellings": ["تورم اطراف"],
    },
    "itching": {
        "formal": ["حكة"],
        "egyptian": ["هرش", "جلدي بيهرشني", "بحك جلدي"],
        "misspellings": ["حكه", "هرش جامد"],
    },
    "skin_rash": {
        "formal": ["طفح جلدي"],
        "egyptian": ["حساسية جلد", "بقع على الجلد", "طفح في الجلد"],
        "misspellings": ["طفح", "حساسيه جلد"],
    },
    "nodal_skin_eruptions": {
        "formal": ["حبوب جلدية", "نتوءات جلدية"],
        "egyptian": ["حبوب طالعة", "بثور صغيرة", "حبوب في الجلد", "حبوب في الوجه"],
        "misspellings": ["حبوب جلد", "نتوءات"],
    },
    "red_spots_over_body": {
        "formal": ["بقع حمراء بالجسم"],
        "egyptian": ["نقط حمرا في جسمي"],
        "misspellings": ["بقع حمراء", "نقط حمراء"],
    },
    "dischromic _patches": {
        "formal": ["تغير لون الجلد", "تصبغات جلدية"],
        "egyptian": ["بقع لونها مختلف", "بقع غامقة", "بقع فاتحة"],
        "misspellings": ["تصبغات"],
    },
    "internal_itching": {
        "formal": ["حكة داخلية"],
        "egyptian": ["هرش من جوه", "حكة داخل الجسم"],
        "misspellings": ["حكه داخليه"],
    },
    "pus_filled_pimples": {
        "formal": ["بثور مليئة بالصديد"],
        "egyptian": ["حبوب فيها صديد", "حبوب ملتهبة", "بثور"],
        "misspellings": ["حبوب صديد", "صديد في الحبوب"],
    },
    "blackheads": {
        "formal": ["رؤوس سوداء"],
        "egyptian": ["نقط سودا في الوجه", "رؤوس سوداء", "روس سوداء"],
        "misspellings": ["رؤوس سوده", "نقط سوداء"],
    },
    "scurring": {
        "formal": ["ندبات جلدية", "قشور جلدية"],
        "egyptian": ["اثار حبوب", "قشرة على الجلد"],
        "misspellings": ["قشور", "ندبات"],
    },
    "skin_peeling": {
        "formal": ["تقشر الجلد"],
        "egyptian": ["جلدي بيقشر", "قشرة جلد"],
        "misspellings": ["تقشير جلد"],
    },
    "silver_like_dusting": {
        "formal": ["قشور فضية"],
        "egyptian": ["قشرة فضية", "قشور بيضا"],
        "misspellings": ["قشور بيضاء"],
    },
    "small_dents_in_nails": {
        "formal": ["حفر صغيرة في الأظافر"],
        "egyptian": ["نقر في الضوافر"],
        "misspellings": ["حفر اظافر"],
    },
    "inflammatory_nails": {
        "formal": ["التهاب الأظافر"],
        "egyptian": ["ضوافري ملتهبة"],
        "misspellings": ["التهاب ضوافر"],
    },
    "blister": {
        "formal": ["فقاعة جلدية"],
        "egyptian": ["فقاقيع في الجلد", "فقاعة"],
        "misspellings": ["فقاقيع"],
    },
    "red_sore_around_nose": {
        "formal": ["قرحة حمراء حول الأنف"],
        "egyptian": ["التهاب حوالين الانف"],
        "misspellings": ["قرحة حول الانف"],
    },
    "yellow_crust_ooze": {
        "formal": ["قشرة صفراء مع إفرازات"],
        "egyptian": ["صديد اصفر", "قشرة صفرا"],
        "misspellings": ["افراز اصفر", "إفراز أصفر"],
    },
    "bruising": {
        "formal": ["كدمات"],
        "egyptian": ["بقع زرقا", "كدمة"],
        "misspellings": ["كدمات زرقاء"],
    },
    "anxiety": {
        "formal": ["قلق"],
        "egyptian": ["متوتر", "خايف", "قلبي مقبوض", "قلقان"],
        "misspellings": ["توتر"],
    },
    "depression": {
        "formal": ["اكتئاب"],
        "egyptian": ["مخنوق نفسيا", "حزين طول الوقت"],
        "misspellings": ["اكتئاب", "زهقان"],
    },
    "irritability": {
        "formal": ["عصبية"],
        "egyptian": ["متعصب", "بتضايق بسرعة"],
        "misspellings": ["عصبيه"],
    },
    "mood_swings": {
        "formal": ["تقلبات مزاجية"],
        "egyptian": ["مزاجي بيتغير بسرعة"],
        "misspellings": ["تقلب مزاج"],
    },
    "family_history": {
        "formal": ["تاريخ عائلي", "تاريخ مرضي عائلي"],
        "egyptian": ["عندنا في العيلة", "وراثي"],
        "misspellings": ["تاريخ عائلى"],
    },
    "receiving_blood_transfusion": {
        "formal": ["نقل دم سابق"],
        "egyptian": ["اخدت نقل دم قبل كده"],
        "misspellings": ["نقل دم"],
    },
    "receiving_unsterile_injections": {
        "formal": ["حقن غير معقمة"],
        "egyptian": ["اخدت حقنة مش مضمونة"],
        "misspellings": ["حقن ملوثة"],
    },
    "history_of_alcohol_consumption": {
        "formal": ["تاريخ شرب كحول"],
        "egyptian": ["بشرب كحول", "شربت خمور"],
        "misspellings": ["كحول", "خمور"],
    },
    "toxic_look_(typhos)": {
        "formal": ["مظهر تسممي شديد"],
        "egyptian": ["شكلي مرهق جدا", "تعبان جدا مع حرارة"],
        "misspellings": ["مظهر تسممي"],
    },
    "fluid_overload": {
        "formal": ["زيادة سوائل بالجسم"],
        "egyptian": ["احتباس سوائل", "جسمي وارم"],
        "misspellings": ["احتباس سوائل"],
    },
    "fluid_overload.1": {
        "formal": ["زيادة سوائل بالجسم"],
        "egyptian": ["احتباس سوائل", "تورم عام"],
        "misspellings": ["احتباس سوائل"],
    },
    "acute_liver_failure": {
        "formal": ["فشل كبدي حاد"],
        "egyptian": ["اصفرار شديد مع تدهور وعي"],
        "misspellings": ["فشل كبد"],
    },
    "ulcers_on_tongue": {
        "formal": ["قرح على اللسان", "تقرحات اللسان"],
        "egyptian": ["لساني فيه قرح", "قرحة في لساني", "لساني مجروح"],
        "misspellings": ["قرح اللسان", "تقرحات لسان"],
    },
    "swelled_lymph_nodes": {
        "formal": ["تورم الغدد الليمفاوية", "تضخم الغدد الليمفاوية"],
        "egyptian": ["غدد رقبتي وارمة", "غدد تحت الفك وارمة"],
        "misspellings": ["تورم الغدد", "تضخم الغدد"],
    },
    "cramps": {
        "formal": ["تقلصات", "تشنجات عضلية"],
        "egyptian": ["شد عضلي", "مغص وتقلصات", "كرامب"],
        "misspellings": ["تقلصات", "شد عضلى"],
    },
    "drying_and_tingling_lips": {
        "formal": ["جفاف وتنميل الشفاه"],
        "egyptian": ["شفايفي ناشفة", "تنميل في الشفايف", "شفايفي بتنمل"],
        "misspellings": ["جفاف الشفايف", "تنميل شفايف"],
    },
    "knee_pain": {
        "formal": ["ألم الركبة", "الم الركبة"],
        "egyptian": ["ركبتي واجعاني", "وجع ركبة"],
        "misspellings": ["الم ركبه", "وجع ركبه"],
    },
    "hip_joint_pain": {
        "formal": ["ألم مفصل الورك", "الم مفصل الورك"],
        "egyptian": ["وجع في الحوض", "مفصل الحوض واجعني"],
        "misspellings": ["الم الورك", "وجع الورك"],
    },
    "swelling_joints": {
        "formal": ["تورم المفاصل"],
        "egyptian": ["مفاصلي وارمة", "تورم في الركبة", "تورم في المفصل"],
        "misspellings": ["تورم مفاصل"],
    },
    "movement_stiffness": {
        "formal": ["تيبس الحركة", "تيبس المفاصل"],
        "egyptian": ["حركتي تقيلة", "مش قادر احرك المفصل", "المفصل ناشف"],
        "misspellings": ["تيبس حركه", "تيبس مفاصل"],
    },
}


PHASE2_TARGETED_SYNONYMS: dict[str, dict[str, list[str]]] = {
    "high_fever": {
        "formal": ["حرارة", "حمى", "حرارة مستمرة", "ارتفاع حرارة"],
        "egyptian": ["سخونية", "حرارة بعد سفر", "حرارة مع صداع", "سخونية شديدة"],
        "misspellings": ["حراره", "سخونيه", "حراة"],
    },
    "sweating": {
        "formal": ["عرق", "تعرق شديد"],
        "egyptian": ["بعرق", "بعرق كتير", "عرقان", "عرق بالليل", "تعرق بالليل"],
        "misspellings": ["عرق كتير", "بعرك"],
    },
    "shivering": {
        "formal": ["رعشة", "رجفة"],
        "egyptian": ["برتعش", "جسمي بيرتعش"],
        "misspellings": ["رعشه", "رجفه"],
    },
    "fatigue": {
        "formal": ["إعياء", "اعياء", "تعب شديد"],
        "egyptian": ["هلكان", "مش قادر اقف", "مش قادر اقوم", "تعبان جدا", "حاسس بهبوط"],
        "misspellings": ["اعيا", "هبوط عام"],
    },
    "lethargy": {
        "formal": ["خمول شديد"],
        "egyptian": ["خمول", "عايز انام طول الوقت", "عاوز انام طول الوقت", "نايم طول الوقت"],
        "misspellings": ["خمولان"],
    },
    "malaise": {
        "formal": ["تعب عام", "إعياء عام", "اعياء عام"],
        "egyptian": ["حاسس بتعب عام", "مش قادر اوصف", "حاسس اني مش طبيعي", "مش طبيعي"],
        "misspellings": ["تعب عام"],
    },
    "loss_of_appetite": {
        "formal": ["فقدان شهية"],
        "egyptian": ["مليش نفس للاكل", "ماليش نفس للاكل", "نفسي مسدودة", "مش عايز اكل"],
        "misspellings": ["مليش نفس للأكل", "فقدان شهيه"],
    },
    "weight_loss": {
        "formal": ["نقص وزن بدون سبب", "نقص وزن غير مفسر"],
        "egyptian": ["بخس", "نقص وزن من غير سبب واضح", "وزني بيقل", "خسيت من غير سبب"],
        "misspellings": ["وزنى بيقل", "نقصان وزن"],
    },
    "irregular_sugar_level": {
        "formal": ["اضطراب السكر", "تذبذب السكر"],
        "egyptian": ["لخبطة في السكر", "السكر ملخبط", "هبوط سكر", "السكر بينزل"],
        "misspellings": ["لخبطه السكر", "لخبطة السكر"],
    },
    "polyuria": {
        "formal": ["تبول متكرر", "تبول ليلي"],
        "egyptian": ["كثرة تبول", "تبول بالليل", "بتبول بالليل", "بدخل الحمام بالليل", "عطش شديد"],
        "misspellings": ["كتره تبول", "تبول كتير"],
    },
    "excessive_hunger": {
        "formal": ["جوع زائد"],
        "egyptian": ["جعان", "جوع شديد", "جعان جدا", "جوع غير طبيعي"],
        "misspellings": ["جعان اوي", "جعان قوى"],
    },
    "enlarged_thyroid": {
        "formal": ["تضخم الغدة", "تورم الغدة"],
        "egyptian": ["رقبتي فيها تورم من الغدة", "رقبتي وارمة من الغدة", "الغدة وارمة"],
        "misspellings": ["تورم الغده", "تضخم الغده"],
    },
    "puffy_face_and_eyes": {
        "formal": ["انتفاخ الوجه", "انتفاخ العين"],
        "egyptian": ["وشي منفخ", "عيني منفخة", "وشي وارم", "عيني وارمة", "تورم في الوجه"],
        "misspellings": ["وشى منفخ", "عينى منفخه"],
    },
    "burning_micturition": {
        "formal": ["حرقان البول", "ألم حارق أثناء التبول"],
        "egyptian": ["حرقان بول", "البول بيحرق", "حرقان وانا بتبول", "بول حارق"],
        "misspellings": ["حرقه بول", "حرقة بول"],
    },
    "spotting_ urination": {
        "formal": ["دم في البول", "نزول دم مع البول"],
        "egyptian": ["نقط دم مع البول", "بول بدم", "الدم في البول"],
        "misspellings": ["دم ف البول", "دم بالبول"],
    },
    "foul_smell_of urine": {
        "formal": ["رائحة كريهة للبول"],
        "egyptian": ["ريحة البول وحشة", "ريحة بول وحشة", "ريحة البول كريهة"],
        "misspellings": ["ريحه البول وحشه", "ريحة بول كريهه"],
    },
    "continuous_feel_of_urine": {
        "formal": ["رغبة مستمرة في التبول"],
        "egyptian": ["حاسس عايز اتبول طول الوقت", "عايز اتبول طول الوقت", "حاسة عايزة اتبول طول الوقت"],
        "misspellings": ["عايز اتبول كتير"],
    },
    "bladder_discomfort": {
        "formal": ["ألم المثانة", "ضغط في المثانة"],
        "egyptian": ["وجع مثانة", "الم مثانة", "وجع تحت البطن مع البول"],
        "misspellings": ["وجع مثانه"],
    },
    "back_pain": {
        "formal": ["ألم الخاصرة", "ألم في الخاصرة"],
        "egyptian": ["الم في الجنب", "وجع في الجنب", "جنبي واجعني"],
        "misspellings": ["الم الجنب", "وجع الجنب"],
    },
    "dark_urine": {
        "formal": ["بول غامق", "بول داكن"],
        "egyptian": ["البول غامق", "بول لونه غامق"],
        "misspellings": ["بول غامء"],
    },
    "yellowing_of_eyes": {
        "formal": ["اصفرار العين", "اصفرار بياض العين"],
        "egyptian": ["اصفرار عين", "صفار في العين", "عيني صفرا"],
        "misspellings": ["صفار العين"],
    },
    "yellowish_skin": {
        "formal": ["اصفرار الجلد", "يرقان"],
        "egyptian": ["جلدي اصفر", "وشي اصفر"],
        "misspellings": ["اصفرار جلد"],
    },
    "dehydration": {
        "formal": ["جفاف شديد", "قلة البول مع جفاف"],
        "egyptian": ["بوقي ناشف", "ريقي ناشف", "عطشان جدا", "مش بتبول"],
        "misspellings": ["بؤي ناشف", "مش بتبول"],
    },
    "stomach_bleeding": {
        "formal": ["قيء دم", "نزيف من المعدة"],
        "egyptian": ["بستفرغ دم", "استفرغ دم", "بترجع دم", "ترجيع دم", "براز اسود"],
        "misspellings": ["بستفرغ دم", "برجع دم"],
    },
    "coma": {
        "formal": ["إغماء", "فقدان وعي"],
        "egyptian": ["اغماء", "اغماء مفاجئ", "فقدان وعي", "وقعت ومش فاكر"],
        "misspellings": ["اغما", "فقدان وعى"],
    },
    "altered_sensorium": {
        "formal": ["تشوش وعي", "ارتباك شديد"],
        "egyptian": ["مش فاكر اللي حصل", "مش فاكرة اللي حصل", "ملخبط ومش فاهم"],
        "misspellings": ["مش فاكر الى حصل"],
    },
    "cramps": {
        "formal": ["تشنجات", "نوبة تشنج"],
        "egyptian": ["تشنجات وفقدان وعي", "جالي تشنج"],
        "misspellings": ["تشنجت"],
    },
    "weakness_of_one_body_side": {
        "formal": ["اعوجاج الوجه", "ميلان الفم"],
        "egyptian": ["وشي معوج", "وشي مايل", "بقي معوج", "الفم معوج"],
        "misspellings": ["وشى معوج", "وشي معووج"],
    },
    "slurred_speech": {
        "formal": ["ثقل الكلام", "تلعثم مفاجئ"],
        "egyptian": ["كلامي تقيل", "لساني تقيل", "مش عارف اتكلم"],
        "misspellings": ["كلامى تقيل"],
    },
    "stiff_neck": {
        "formal": ["تيبس الرقبة", "تيبس رقبة"],
        "egyptian": ["رقبتي ناشفة", "مش قادر احرك رقبتي"],
        "misspellings": ["تيبس رقبه"],
    },
}


def flatten_symptom_synonyms(
    grouped: dict[str, dict[str, list[str]]] = ARABIC_SYMPTOM_SYNONYMS,
) -> dict[str, list[str]]:
    combined: dict[str, dict[str, list[str]]] = {key: dict(value) for key, value in grouped.items()}
    for symptom, groups in PHASE2_TARGETED_SYNONYMS.items():
        target = combined.setdefault(symptom, {})
        for group_name, phrases in groups.items():
            target.setdefault(group_name, [])
            target[group_name].extend(phrases)

    flattened: dict[str, list[str]] = {}
    for symptom, groups in combined.items():
        phrases: list[str] = []
        for values in groups.values():
            phrases.extend(values)
        flattened[symptom] = list(dict.fromkeys(phrases))
    return flattened
