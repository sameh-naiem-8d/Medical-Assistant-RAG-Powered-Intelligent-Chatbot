from __future__ import annotations


DIAGNOSIS_DISPLAY_AR: dict[str, str] = {
    "(vertigo) Paroymsal  Positional Vertigo": "دوخة/دوار يحتاج تقييم",
    "AIDS": "نقص المناعة المكتسب",
    "Acne": "حب الشباب",
    "Alcoholic hepatitis": "التهاب كبدي مرتبط بالكحول",
    "Allergy": "حساسية",
    "Arthritis": "التهاب في المفاصل",
    "Bronchial Asthma": "ربو شعبي",
    "Cervical spondylosis": "خشونة فقرات الرقبة / مشكلة في فقرات الرقبة",
    "Chicken pox": "جدري الماء",
    "Chronic cholestasis": "ركود صفراوي مزمن",
    "Common Cold": "نزلة برد / التهاب في الجهاز التنفسي العلوي",
    "Viral or flu-like illness": "عدوى فيروسية / دور برد أو إنفلونزا محتملة",
    "General symptoms needing clarification": "أعراض عامة تحتاج أسئلة توضيحية",
    "Dengue": "حمى الضنك",
    "Diabetes": "السكري",
    "Dimorphic hemmorhoids(piles)": "بواسير",
    "Drug Reaction": "تفاعل أو حساسية من دواء",
    "Fungal infection": "عدوى فطرية جلدية",
    "GERD": "ارتجاع المريء",
    "Gastroenteritis": "التهاب المعدة والأمعاء",
    "Heart attack": "اشتباه مشكلة قلبية طارئة",
    "Neurological emergency concern": "أعراض عصبية خطيرة محتملة",
    "Cardiac emergency concern": "اشتباه مشكلة قلبية طارئة",
    "Pregnancy emergency concern": "أعراض حمل طارئة محتملة",
    "Poisoning emergency concern": "اشتباه تسمم أو جرعة زائدة",
    "Severe allergic reaction concern": "اشتباه حساسية شديدة أو تورم تحسسي",
    "Head injury emergency concern": "إصابة رأس تحتاج طوارئ",
    "Eye emergency concern": "أعراض عين طارئة محتملة",
    "Respiratory emergency concern": "أعراض تنفسية خطيرة محتملة",
    "Severe dehydration or infection concern": "اشتباه جفاف شديد أو عدوى خطيرة",
    "Diabetes emergency concern": "اشتباه اضطراب سكر خطير",
    "Abdominal surgical emergency concern": "ألم بطن خطير يحتاج طوارئ",
    "Urinary or kidney emergency concern": "اشتباه مشكلة كلى أو مسالك طارئة",
    "Pediatric emergency concern": "أعراض خطيرة عند طفل تحتاج طوارئ",
    "Self-harm emergency concern": "خطر إيذاء النفس يحتاج مساعدة عاجلة",
    "hepatitis A": "التهاب الكبد أ",
    "Hepatitis B": "التهاب الكبد ب",
    "Hepatitis C": "التهاب الكبد ج",
    "Hepatitis D": "التهاب الكبد د",
    "Hepatitis E": "التهاب الكبد هـ",
    "Hypertension": "ارتفاع ضغط الدم",
    "Hyperthyroidism": "فرط نشاط الغدة الدرقية",
    "Hypoglycemia": "انخفاض سكر الدم",
    "Hypothyroidism": "قصور الغدة الدرقية",
    "Impetigo": "عدوى جلدية سطحية",
    "Jaundice": "يرقان / اصفرار",
    "Malaria": "ملاريا",
    "Migraine": "صداع نصفي",
    "Osteoarthristis": "خشونة المفاصل",
    "Paralysis (brain hemorrhage)": "أعراض عصبية طارئة تحتاج تقييم",
    "Peptic ulcer diseae": "مشكلة/التهاب أو قرحة بالمعدة",
    "Pneumonia": "التهاب رئوي",
    "Psoriasis": "صدفية",
    "Tuberculosis": "درن رئوي",
    "Typhoid": "حمى التيفود",
    "Urinary tract infection": "التهاب في المسالك البولية",
    "Varicose veins": "دوالي",
    "General medical evaluation": "أعراض عامة تحتاج تقييم طبي",
}


DOCTOR_DISPLAY_AR: dict[str, str] = {
    "General Practitioner": "طبيب عام / باطنة",
    "Pulmonologist": "طبيب صدر",
    "Cardiologist": "طبيب قلب",
    "Endocrinologist": "طبيب غدد صماء",
    "Neurologist": "طبيب مخ وأعصاب",
    "Gastroenterologist": "طبيب جهاز هضمي / باطنة",
    "Urologist": "طبيب مسالك بولية",
    "Dermatologist": "طبيب جلدية",
    "Allergist": "طبيب حساسية",
    "Infectious disease specialist": "طبيب أمراض معدية / باطنة",
    "Emergency care": "الطوارئ فورًا",
    "ENT specialist": "طبيب أنف وأذن وحنجرة",
    "Pediatrician": "طبيب أطفال",
    "Gynecologist": "طبيب/طوارئ نساء وتوليد حسب شدة الحالة",
    "Emergency Department / Neurologist": "الطوارئ فورًا / طبيب مخ وأعصاب",
    "Emergency Department / Cardiologist": "الطوارئ فورًا / طبيب قلب",
    "Emergency Department / Gynecologist": "طوارئ نساء وتوليد فورًا",
    "Poison Control / Emergency Department": "الطوارئ أو مركز السموم فورًا",
    "Emergency Department / Allergy or Pulmonology": "الطوارئ فورًا / حساسية ومناعة أو صدر حسب الحالة",
    "Emergency Department / Neurology or Surgery": "الطوارئ فورًا / مخ وأعصاب أو جراحة",
    "Emergency Department / Ophthalmologist": "الطوارئ فورًا / طبيب عيون",
    "Emergency Department / Pulmonologist": "الطوارئ فورًا / طبيب صدر",
    "Emergency Department / Internal Medicine": "الطوارئ فورًا / باطنة",
    "Emergency Department / Internal Medicine or Endocrinologist": "الطوارئ فورًا / باطنة أو غدد صماء",
    "Emergency Department / Surgery or Gastroenterologist": "الطوارئ فورًا / جراحة أو جهاز هضمي",
    "Emergency Department / Urologist": "الطوارئ فورًا / مسالك بولية",
    "Pediatric Emergency Department": "طوارئ أطفال فورًا",
    "Emergency Department / Psychiatrist": "الطوارئ فورًا / طبيب نفسي",
    "Psychiatrist": "طبيب نفسي",
    "Dentist": "طبيب أسنان",
    "Ophthalmologist": "طبيب عيون",
    "Orthopedic doctor": "طبيب عظام",
    "Needs more information": "يحتاج معلومات أكثر",
    "Not needed": "غير مطلوب حاليًا",
}


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06FF" for char in text)


def display_diagnosis_ar(diagnosis: str | None) -> str | None:
    if not diagnosis:
        return None
    if diagnosis in DIAGNOSIS_DISPLAY_AR:
        return DIAGNOSIS_DISPLAY_AR[diagnosis]
    if _contains_arabic(diagnosis):
        return diagnosis
    return "حالة تحتاج تقييم طبي"


def display_doctor_ar(doctor: str | None) -> str | None:
    if not doctor:
        return None
    if doctor in DOCTOR_DISPLAY_AR:
        return DOCTOR_DISPLAY_AR[doctor]
    if _contains_arabic(doctor):
        return doctor
    return "تخصص طبي مناسب"
