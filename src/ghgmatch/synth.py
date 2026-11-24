import random
import string

import numpy as np
import pandas as pd

ITEMS: dict[str, list[str]] = {
    "IT-HW": ["Laptop ThinkPad T14", "Notebook Dell Latitude", "Monitor 27 Zoll", "Bildschirm 4K", "Docking Station USB-C",
              "Server PowerEdge R760", "GPU Nvidia RTX A6000", "Festplatte SSD 2TB", "Tastatur und Maus", "MacBook Pro 14",
              "Netzwerk Switch 24 Port", "Workstation HP Z4"],
    "IT-SW": ["Microsoft 365 Lizenz", "MATLAB annual license", "Adobe Creative Cloud Abo", "AWS cloud credits",
              "Zoom Pro subscription", "Software Wartungsvertrag", "GitHub Enterprise seats", "Overleaf Professional",
              "Altium Designer license renewal", "Cloud hosting Jahresgebuehr"],
    "LAB-CHEM": ["Ethanol absolut 2.5L", "Aceton p.a.", "Salzsaeure 37%", "Natriumchlorid 1kg", "Methanol HPLC grade",
                 "Sulfuric acid 95-97%", "Isopropanol 5L", "Dichlormethan stabilisiert", "Natronlauge 1M",
                 "Acetonitrile LC-MS"],
    "LAB-BIO": ["Antikoerper anti-GFP", "DMEM Medium 500ml", "Fetal Bovine Serum", "Taq Polymerase Kit",
                "RNA Extraction Kit", "Trypsin-EDTA", "Restriktionsenzym EcoRI", "ELISA Kit IL-6", "Primer Synthese",
                "Agarose fuer Gelelektrophorese"],
    "LAB-PLAST": ["Pipettenspitzen 200ul", "Falcon Tubes 50ml", "Eppendorf Tubes 1.5ml", "Zellkulturflaschen T75",
                  "96-well Platten", "Nitril Handschuhe M", "Petrischalen steril", "Serologische Pipetten 10ml",
                  "Filter tips 1000ul", "Kryoroehrchen 2ml"],
    "LAB-EQ": ["Zentrifuge Eppendorf 5424", "Oszilloskop Keysight 4 Kanal", "Mikroskop Objektiv 40x",
               "Pipette Research plus", "Magnetruehrer mit Heizung", "pH-Meter", "Netzgeraet Labor 30V",
               "Spektrometer USB", "Waage Analytik 0.1mg", "Thermocycler PCR"],
    "GAS": ["Stickstoff 5.0 Flasche", "Helium fluessig 100L", "Argon 4.6 Gasflasche", "CO2 Flasche 30kg",
            "Druckgas Miete Flasche", "Fluessigstickstoff Lieferung", "Sauerstoff technisch", "Helium Rueckgewinnung"],
    "ELEC-COMP": ["Widerstand 10k SMD", "Kondensator 100nF", "Mikrocontroller STM32", "Leiterplatte Fertigung PCB",
                  "Steckverbinder JST", "Sensor MEMS Drucksensor", "Kabel geschirmt 10m", "Operationsverstaerker",
                  "FPGA Development Board", "Spannungsregler LDO"],
    "MECH": ["Schrauben M3 Edelstahl", "Aluprofil 30x30", "Kugellager 608", "Werkzeugset", "CNC Fraesteile Alu",
             "Gewindestange M8", "Blech Edelstahl 2mm", "Linearfuehrung", "Zahnriemen GT2", "Drehteile Messing"],
    "FURN": ["Buerostuhl ergonomisch", "Schreibtisch hoehenverstellbar", "Regal Metall", "Rollcontainer",
             "Laborbank", "Aktenschrank", "Besprechungstisch", "Garderobe"],
    "PAPER": ["Kopierpapier A4 80g", "Toner HP 410X", "Post-it Notizen", "Ordner breit", "Briefumschlaege C4",
              "Druck Poster A0", "Kugelschreiber", "Visitenkarten Druck"],
    "AIR": ["Flug ZRH-LHR Economy", "Flight Zurich Boston return", "Swiss flight ticket ZRH-FRA", "Airline Umbuchungsgebuehr",
            "Flug Konferenz San Francisco", "Lufthansa ticket Munich", "easyJet Basel Berlin"],
    "RAIL": ["SBB Billett Zuerich Bern", "Halbtax Abo", "GA Travelcard", "Zugticket Lausanne",
             "Deutsche Bahn Ticket Stuttgart", "SBB Tageskarte", "Tram Ticket ZVV"],
    "HOTEL": ["Hotel 3 Naechte Konferenz", "Uebernachtung Bern", "Hotel booking Boston", "Airbnb Workshop Lausanne",
              "Hotel Zimmer Einzel", "Accommodation Summer School"],
    "FOOD": ["Apero Institut", "Catering Workshop", "Kaffee Bohnen Kaffeemaschine", "Mittagessen Gruppe",
             "Sandwiches Meeting", "Weihnachtsessen Team", "Getraenke Event"],
    "ENERGY": ["Stromrechnung Q3", "Fernwaerme Gebaeude", "Erdgas Heizung", "Electricity supply EWZ",
               "Heizoel Lieferung", "Strom Netznutzung"],
    "CONSTR": ["Umbau Labor Raum", "Malerarbeiten Buero", "Reparatur Lueftung", "Installation Abzug",
               "Sanitaer Reparatur", "Bodenbelag erneuern", "Elektroinstallation Labor"],
    "SERV": ["Uebersetzung Dokument", "Konferenzgebuehr IEEE", "Beratung Rechtsfragen", "Weiterbildung Kurs Python",
             "Registration fee workshop", "Consulting Statistik", "Lektorat Paper"],
}

VENDORS: dict[str, list[str]] = {
    "IT-HW": ["Lenovo", "Dell", "Digitec", "Brack", "Apple", "HP"],
    "IT-SW": ["Microsoft", "MathWorks", "Adobe", "Amazon Web Services", "SoftwareONE"],
    "LAB-CHEM": ["Sigma-Aldrich", "VWR", "Merck", "Thermo Fisher", "Carl Roth"],
    "LAB-BIO": ["Thermo Fisher", "Qiagen", "Abcam", "NEB", "Sigma-Aldrich"],
    "LAB-PLAST": ["Sarstedt", "Eppendorf", "VWR", "Corning", "Greiner"],
    "LAB-EQ": ["Eppendorf", "Keysight", "Zeiss", "Mettler Toledo", "Distrelec"],
    "GAS": ["PanGas", "Carbagas", "Linde", "Messer"],
    "ELEC-COMP": ["Digi-Key", "Mouser", "Distrelec", "Farnell", "Eurocircuits"],
    "MECH": ["Bossard", "Misumi", "Igus", "Hornbach", "ETH Werkstatt"],
    "FURN": ["Vitra", "USM", "Ikea", "Lista Office"],
    "PAPER": ["Lyreco", "Papyrus", "ETH Print+Publish", "Ofrex"],
    "AIR": ["SWISS", "Lufthansa", "easyJet", "BCD Travel"],
    "RAIL": ["SBB", "DB", "ZVV"],
    "HOTEL": ["Booking.com", "Hotel Schweizerhof", "Airbnb", "BCD Travel"],
    "FOOD": ["SV Group", "Coop", "Migros", "Polyterrasse Catering"],
    "ENERGY": ["EWZ", "Energie 360", "Axpo"],
    "CONSTR": ["Implenia", "Immobilien ETH", "Burkhalter", "Walo"],
    "SERV": ["IEEE", "Supertext", "PwC", "ETH Weiterbildung"],
}

SPEND_MEDIAN_CHF: dict[str, float] = {
    "IT-HW": 1800, "IT-SW": 900, "LAB-CHEM": 250, "LAB-BIO": 600, "LAB-PLAST": 300, "LAB-EQ": 3500,
    "GAS": 450, "ELEC-COMP": 150, "MECH": 200, "FURN": 900, "PAPER": 120, "AIR": 1100, "RAIL": 90,
    "HOTEL": 500, "FOOD": 400, "ENERGY": 12000, "CONSTR": 8000, "SERV": 700,
}

GENERALIST_VENDORS = ["Galaxus", "Amazon Business", "Conrad", "Jumbo", "ETH Store"]

ABBREVIATIONS = {"und": "u.", "Labor": "Lab", "Flasche": "Fl.", "Lizenz": "Liz.", "Gebaeude": "Geb.",
                 "Stueck": "Stk", "Kit": "KIT", "Lieferung": "Lief."}


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 2)
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def _sku(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=2)) + "-" + "".join(rng.choices(string.digits, k=rng.randint(4, 7)))


def noisy_description(item: str, vendor: str, rng: random.Random, noise: float) -> str:
    words = item.split()
    words = [ABBREVIATIONS.get(w, w) if rng.random() < noise else w for w in words]
    words = [_typo(w, rng) if rng.random() < noise * 0.4 else w for w in words]
    if rng.random() < noise * 0.5 and len(words) > 2:
        words = words[: rng.randint(2, len(words))]
    parts = [" ".join(words)]
    if rng.random() < 0.5:
        parts.append(vendor)
    if rng.random() < 0.4:
        parts.append(_sku(rng))
    if rng.random() < 0.3:
        parts.append(f"{rng.randint(1, 50)} Stk")
    if rng.random() < noise * 0.3:
        rng.shuffle(parts)
    text = " ".join(parts)
    if rng.random() < noise * 0.5:
        text = text.upper()
    elif rng.random() < noise * 0.3:
        text = text.lower()
    return text


def item_pool(code: str, held_out: int, include_held_out: bool) -> list[tuple[str, bool]]:
    items = ITEMS[code]
    seen = items[: len(items) - held_out]
    novel = items[len(items) - held_out:]
    return [(i, False) for i in seen] + ([(i, True) for i in novel] if include_held_out else [])


def generate(n: int = 3000, noise: float = 0.5, seed: int = 7, held_out: int = 3,
             include_held_out: bool = False, generalist_share: float = 0.2) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    codes = list(ITEMS)
    weights = np.array([1.4 if c.startswith("LAB") else 1.0 for c in codes])
    weights /= weights.sum()
    pools = {c: item_pool(c, held_out, include_held_out) for c in codes}
    rows = []
    for k in range(n):
        code = codes[np_rng.choice(len(codes), p=weights)]
        item, novel = rng.choice(pools[code])
        generalist = code not in {"AIR", "RAIL", "ENERGY", "CONSTR", "GAS"} and rng.random() < generalist_share
        vendor = rng.choice(GENERALIST_VENDORS) if generalist else rng.choice(VENDORS[code])
        spend = float(np.round(np_rng.lognormal(np.log(SPEND_MEDIAN_CHF[code]), 0.8), 2))
        rows.append({
            "line_id": f"PO-{seed}-{k:05d}",
            "description": noisy_description(item, vendor, rng, noise),
            "vendor": vendor,
            "spend_chf": spend,
            "true_code": code,
            "novel_item": novel,
        })
    return pd.DataFrame(rows)
