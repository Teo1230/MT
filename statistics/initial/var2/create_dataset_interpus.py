emoji_to_words = {
    # sport / activitate
    "🏋️‍♀️": ["sport", "antrenament", "sala", "forta", "fitness"],
    "🤸‍♂️": ["gimnastica", "stretching", "miscare"],
    "🏃‍♀️": ["alergare", "cardio", "jogging"],
    "🧘‍♀️": ["yoga", "meditatie", "mindfulness", "relaxare"],
    "👟": ["adidasi", "pantofi_sport", "alergare"],
    "🏄‍♀️": ["surf", "ocean", "sport_de_apa"],
    "🏊‍♂️": ["inot", "piscina", "apa", "sport_de_apa"],
    "⚽": ["fotbal", "minge", "meci", "gol"],
    "🏀": ["baschet", "cos", "teren"],
    "🎾": ["tenis", "racheta", "serva"],
    "🏈": ["fotbal_american", "meci", "echipa"],
    "🤽‍♀️": ["volei_pe_apa", "piscina", "echipa"],

    # emoții / stări
    "💔": ["inima_franta", "durere", "suferinta", "dezamagire"],
    "❤️": ["iubire", "dragoste", "inima", "afectiune"],
    "😢": ["tristete", "lacrimi", "plang"],
    "😟": ["ingrijorare", "anxietate", "emotii"],
    "😌": ["liniste", "calm", "relaxare", "multumire"],
    "😡": ["furie", "nerabdare", "frustrare"],
    "🥺": ["rugaminte", "trist", "sensibil"],
    "😍": ["adoratie", "iubire", "incantare"],
    "😁": ["zambet_mare", "fericire", "bucurie"],
    "😊": ["zambet", "recunostinta", "dragalas"],
    "😄": ["bucurie", "ras", "fericire"],
    "😮": ["uimire", "surpriza"],
    "🤩": ["extaz", "uimit", "wow"],
    "🥰": ["dragalasenie", "iubire", "caldura"],
    "🤬": ["nervi", "furie", "frustrare"],

    # prietenie / oameni
    "👥": ["prieteni", "grup", "oameni"],
    "👯‍♀️": ["prietene", "distractie", "dans"],
    "👨‍👩‍👧‍👦": ["familie", "parinti", "copii"],
    "👧": ["copil", "fetita"],
    "👩‍👧": ["mama", "copil", "familie"],
    "👨‍🦳": ["bunic", "batran", "intelepciune"],
    "👵": ["bunica", "batrana", "familie"],
    "👭": ["prietene", "prietene_apropiate"],
    "👥": ["prieteni", "grup", "social"],

    # natura / vreme
    "🌞": ["soare", "zi_frumoasa", "vara", "caldura"],
    "☀️": ["soare", "lumina", "caldura"],
    "🌅": ["rasarit", "apus", "orizont"],
    "🌄": ["rasarit", "munte", "peisaj"],
    "🌻": ["floarea_soarelui", "galben", "vara"],
    "🌼": ["floare", "primavara"],
    "🌸": ["flori", "cireș", "primavara"],
    "🌺": ["flori", "hibiscus", "tropical"],
    "🌷": ["lalea", "primavara"],
    "🍃": ["frunze", "vant_usor", "natura"],
    "🌿": ["natura", "verde", "plante"],
    "🌊": ["ocean", "valuri", "apa", "mare",],
    "🌧️": ["ploaie", "nor", "vreme_urata"],
    "☔": ["umbrela", "ploaie"],
    "❄️": ["zapada", "iarna", "frig"],
    "⛈️": ["furtuna", "tunete", "fulgere"],
    "⚡": ["fulger", "energie", "putere"],

    # animale
    "🐼": ["panda", "bambus", "lenes", "dragalas"],
    "🐘": ["elefant", "memorie", "inteligență", "turma"],
    "🐧": ["pinguin", "frig", "gheata", "mers_caraghios"],
    "🐬": ["delfin", "ocean", "inteligent", "jucaus"],
    "🦥": ["lenes", "incet", "relaxare"],
    "🐝": ["albina", "miere", "polen", "flori"],
    "🦋": ["fluture", "transformare", "delicatete"],

    # haine / stil
    "👖": ["blugi", "pantaloni", "jeans"],
    "👕": ["tricou", "top", "casual"],
    "🧥": ["jacheta", "haina", "toamna"],
    "🧦": ["sosete", "picioare_calde"],
    "👗": ["rochie", "elegant", "feminin"],
    "🩲": ["lenjerie", "boxeri"],
    "👠": ["pantofi_cu_toc", "elegant"],
    "🧣": ["fular", "iarna", "cald"],
    "🧥": ["jacheta", "pulover", "frig"],
    "👟": ["adidasi", "incaltaminte_sport"],

    # mâncare / băutură
    "🍟": ["cartofi_prajiti", "fast_food", "gustare"],
    "🍔": ["burger", "fast_food", "sandvis"],
    "🍕": ["pizza", "cina", "junk_food"],
    "🍣": ["sushi", "japonez", "cina"],
    "🍞": ["paine", "felie", "pranz"],
    "🍔": ["hamburger", "cina", "gustare"],
    "🍕": ["pizza", "cina", "toppinguri"],
    "🍹": ["cocktail", "bautura", "vara"],
    "🍷": ["vin", "cina", "romantic"],
    "🍇": ["struguri", "fructe"],
    "🍓": ["capsuni", "fructe"],
    "🥭": ["mango", "fruct_tropical"],

    # unelte / DIY
    "🛠️": ["unelte", "reparatii", "diy", "constructii"],
    "🔧": ["cheie_fixa", "reparatii"],
    "🔨": ["ciocan", "bataie_cuie"],
    "🪚": ["fierastrau", "taiere_lemn"],
    "🔌": ["priza", "electric", "alimentare"],
    "⚙️": ["mecanism", "roti_dintate", "sistem"],
    "📏": ["rigla", "masurare"],
    "📐": ["echer", "unghiuri", "geometrie"],
    "🪜": ["scara", "urcat"],
    "🪵": ["lemn", "scandura"],

    # tehnologie / muncă
    "💻": ["laptop", "computer", "munca_remote", "online"],
    "📱": ["telefon", "smartphone", "mesaje", "apel"],
    "📸": ["poza", "camera", "fotografie"],
    "📺": ["televizor", "film", "serial"],
    "🖨️": ["imprimanta", "print"],
    "🗂️": ["organizare", "documente", "dosare"],

    # calatorii / locuri
    "🚗": ["masina", "drum", "calatorie"],
    "🚙": ["suv", "masina_mare", "drum_lung"],
    "🏍️": ["motocicleta", "libertate", "drum_deschis"],
    "✈️": ["avion", "zbor", "calatorie"],
    "🏝️": ["insula", "plaja", "paradis"],
    "🏖️": ["plaja", "vacanta", "mare"],
    "🏔️": ["munte", "stanca", "altitudine"],
    "🏕️": ["camping", "cort", "aventura"],
    "🏛️": ["arhitectura_clasica", "muzeu", "istorie"],
    "🏰": ["castel", "fortareata", "istoric"],
    "🕌": ["moschee", "religie", "arhitectura"],
    "🕍": ["sinagoga", "cladire_istorica"],
    "🇨🇦": ["canada", "tara", "nord_america"],
    "🇮🇹": ["italia", "roma", "paste", "pizza"],
    "🇯🇵": ["japonia", "tokyo", "sushi", "anime"],
    "🇮🇸": ["islanda", "gheata", "vulcani"],
    "🇨🇭": ["elvetia", "munti", "ciocolata"],

    # arta / muzica / creativ
    "🎨": ["pictura", "arta", "culori"],
    "🖼️": ["tablou", "cadru", "expozitie"],
    "🎶": ["muzica", "melodie"],
    "🎸": ["chitara", "rock", "instrument"],
    "🎤": ["microfon", "concert", "cantat"],
    "🎻": ["vioara", "clasic"],
    "🎭": ["teatru", "actorie", "scena"],
    "📖": ["carte", "citit", "poveste"],

    # casa / confort
    "🏠": ["casa", "acasa", "locuinta"],
    "🛏️": ["pat", "somn"],
    "🛋️": ["canapea", "living"],
    "🛌": ["somn", "odihna"],
    "🛁": ["baie", "relaxare"],
    "🧺": ["cos_rufe", "organizare"],
    "🧼": ["sapun", "curatenie"],

    # recunoștință / spiritual
    "🙏": ["recunostinta", "multumire", "rugaciune"],
    "🌈": ["curcubeu", "speranta"],
    "✨": ["magie", "stralucire", "inspirație"],
    "🌟": ["stea", "special", "remarcabil"],

    # sculptură / artă / cer
    "🗿": ["sculptura", "piatra", "statue"],
    "🪞": ["oglinda", "reflexie", "imagine"],
    "🌌": ["cer_instelat", "cosmos", "noapte"],

    # cinema
    "🎬": ["film", "clapeta", "regie"],
    "🎥": ["camera_video", "filmare"],
    "🍿": ["popcorn", "cinema", "snack"],
    "🎟️": ["bilet", "intrare", "spectacol"],
    "💺": ["scaun", "loc", "cinema"],
    "🎞️": ["pelicula", "film_vechi"],
    "🍫": ["ciocolata", "dulce", "snack"],

    # clădiri / timp
    "⛪": ["catedrala", "biserica", "religie"],
    "⏳": ["nisiparnita", "timp", "asteptare"],

    # gătit / desert / bucătar
    "🍰": ["tort", "desert"],
    "🥧": ["placinta", "desert"],
    "🍪": ["biscuit", "cookie", "gustare"],
    "🍝": ["paste", "spaghete"],
    "👨‍🍳": ["bucatar", "chef", "gatit"],

    # powerlifting / energie
    "🏋️‍♂️": ["ridicare_greutati", "sala", "forta"],
    "💥": ["explozie", "energie", "impact"],
    "⛏️": ["tarnacop", "minerit", "sapare"],
    "🔘": ["buton", "selectie"],
    "❣️": ["inima_exclamatie", "iubire_puternica"],
    "🥤": ["pahar_cu_pai", "bautura", "suc"],
    "💨": ["viteza", "fuga", "rapid"],

    # karaoke / jocuri / copii
    "😂": ["ras", "amuzant", "haha"],
    "🏘️": ["cartier", "blocuri", "case"],
    "🎮": ["gamepad", "joc_video"],
    "👶": ["bebelus", "copil_mic"],
    "👍": ["like", "aprobare", "bine"],

    # natură / fermă
    "🌳": ["copac", "padure"],
    "👧🏻": ["fetita", "copil", "nepoata"],
    "🍀": ["trifoi", "noroc"],
    "🧒🏽": ["copil", "baietel"],
    "🌍": ["pamant", "planeta", "lume"],
    "🏃🏽‍♂️": ["alergator", "fuga"],
    "🔍": ["lupa", "cautare"],
    "🔭": ["telescop", "stele"],
    "🆕": ["nou", "recent"],
    "🗺️": ["harta", "calatorie"],
    "🚜": ["tractor", "ferma"],
    "🌾": ["grau", "camp", "recolta"],
    "🐄": ["vaca", "ferma"],
    "💛": ["inima_galbena", "prietenie", "caldura"],
    "🥗": ["salata", "mancare_sanatoasa"],
    "🌱": ["lastar", "planta_tanara"],

    # relații / shopping / emoții
    "🛍️": ["cumparaturi", "shopping", "sacose"],
    "♥️": ["inima_rosie", "iubire"],
    "💕": ["iubire_dubla", "dragalasenie"],
    "📚": ["carti", "studiu", "biblioteca"],
    "☕": ["cafea", "ceai", "pauza"],

    # acvariu / mare
    "🐠": ["peste_tropical", "acvariu"],
    "🐳": ["balena", "mare", "ocean"],
    "💙": ["inima_albastra", "ocean", "loialitate"],

    # ciocan / unelte puternice
    "🔨": ["ciocan", "cuie"],
    "🪚": ["fierastrau", "taiere_lemn"],
    "🏢": ["cladire_birouri", "bloc"],
    "🧥": ["jacheta", "haina"],

    # lei / animale sălbatice
    "🦁": ["leu", "rege_jungla", "curaj"],

    # alergare dimineața
    "🏃‍♂️": ["alergare", "fuga"],
    "🌤️": ["soare_cu_nori", "vreme_placuta"],

    # animale de companie
    "🐶": ["caine", "animal_de_companie"],
    "🐾": ["urme_labute", "animale"],

    # modă
    "👠": ["tocuri", "pantofi_eleganti"],
    "🖤": ["inima_neagra", "stil", "rock"],
    "⌚": ["ceas", "timp"],
    "🏖️": ["plaja", "sezlong", "mare"],

    # lux / masini sport
    "💎": ["diamant", "lux", "stralucire"],
    "🏎️": ["masina_sport", "viteza"],
    "🌠": ["stea_cazatoare", "dorinta"],
    "⚡": ["energie", "fulger", "putere"],

    # motivație / muncă / productivitate
    "🏀": ["baschet", "meci"],
    "🐯": ["tigrul", "putere", "agilitate"],
    "👩🏽": ["femeie", "ten_inchis"],
    "🧣": ["fular", "iarna"],
    "🌡️": ["temperatura", "febra", "caldura"],
    "📈": ["grafic_crestere", "progres"],
    "🧰": ["trusa_unelte", "scaula"],
    "⏰": ["ceas_deșteptător", "alarma"],

    # emoții suplimentare
    "😔": ["tristete", "melancolie"],
    "😩": ["epuizare", "oboseala"],
    "😓": ["transpiratie", "stres"],

    # iarnă / vreme
    "⛄": ["om_de_zapada"],
    "🌫️": ["ceata", "vizibilitate_scazuta"],

    # flori / girafe / animale
    "🎈": ["balon", "petrecere"],
    "🦒": ["girafa", "gat_lung"],
    "🐆": ["ghepard", "viteza"],

    # bucătărie / electrocasnice
    "🍳": ["oua", "tigaie"],
    "🥘": ["tocanita", "mancare_gatita"],
    "🧪": ["experiment", "chimie"],
    "🧹": ["matura", "curatenie"],
    "🥫": ["conserva", "mancare_la_borcan"],

    # comunicare / scris / creativ
    "📝": ["notite", "scris"],
    "💬": ["mesaj", "conversatie"],
    "✏️": ["creion", "schita"],

    # locuri / arhitectură / cultură
    "🏛️": ["cladire_clasica", "institutie"],
    "🏯": ["castel_japonez", "templu"],
    "🛕": ["templu_indian"],
    "🕋": ["kaaba", "loc_sfânt"],
    "🗺️": ["harta", "calatorie"],
    "🏙️": ["oras", "skyline"],
    "🏣": ["oficiu_postal"],
    "🏟️": ["stadion", "arena"],

    # stele / recunoștință / emotiv
    "🌟": ["stea", "stralucire"],
    "😭": ["plans", "lacrimi"],
}


import csv
import re

# ------------------------------
# CONFIG
# ------------------------------

INPUT_CSV = "text2emoji_ro_valid_clean.csv"
OUTPUT_CSV = "text2emoji_ro_valid_clean_task2.csv"
TEXT_COLUMN = "ro"
# ------------------------------
# CONVERSIE: cuvânt -> emoji
# ------------------------------

word_to_emoji = {}

for emoji, words in emoji_to_words.items():
    for w in words:
        word_to_emoji[w.lower()] = emoji


# ------------------------------
# INLOCUIRE CUVINTE CU EMOJI
# ------------------------------

def replace_words(text, mapping):
    if not text:
        return text

    new_text = text

    # întâi cuvintele/expresiile mai lungi
    items = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)

    for word, emoji in items:
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", flags=re.IGNORECASE)
        new_text = pattern.sub(emoji, new_text)

    return new_text


# ------------------------------
# PROCESARE CSV
# ------------------------------

rows = []

with open(INPUT_CSV, "r", encoding="utf-8", newline="") as fin:
    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames

    if TEXT_COLUMN not in fieldnames:
        raise ValueError(f"Coloana '{TEXT_COLUMN}' nu există în CSV. Ai coloanele: {fieldnames}")

    for row in reader:
        original = row.get(TEXT_COLUMN, "")
        replaced = replace_words(original, word_to_emoji)
        row[TEXT_COLUMN] = replaced    # 🔥 suprascriem direct coloana
        rows.append(row)


# Scriem CSV-ul final
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

