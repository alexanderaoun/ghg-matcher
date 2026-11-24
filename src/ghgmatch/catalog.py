from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Category:
    code: str
    name: str
    description: str
    kg_co2e_per_chf: float


CATALOG: tuple[Category, ...] = (
    Category("IT-HW", "Computers and electronic equipment",
             "Manufacture of computers, laptops, servers, monitors, peripherals, network and storage hardware", 0.42),
    Category("IT-SW", "Software and IT services",
             "Software licences, cloud computing, hosting, subscriptions and IT consulting services", 0.08),
    Category("LAB-CHEM", "Laboratory chemicals",
             "Basic and fine chemicals, solvents, reagents, acids, salts and analytical standards for laboratories", 0.61),
    Category("LAB-BIO", "Biological reagents and consumables",
             "Antibodies, enzymes, cell culture media, sera, kits and other life science reagents", 0.35),
    Category("LAB-PLAST", "Plastic laboratory consumables",
             "Disposable plastic labware such as pipette tips, tubes, flasks, plates and gloves", 0.55),
    Category("LAB-EQ", "Scientific instruments and lab equipment",
             "Measuring, analytical and optical instruments, microscopes, centrifuges, oscilloscopes and lab devices", 0.30),
    Category("GAS", "Industrial and laboratory gases",
             "Compressed and liquefied gases such as nitrogen, helium, argon, carbon dioxide and gas cylinders", 0.95),
    Category("ELEC-COMP", "Electronic components",
             "Integrated circuits, sensors, resistors, capacitors, connectors, printed circuit boards and cables", 0.48),
    Category("MECH", "Metal products and mechanical parts",
             "Fabricated metal products, screws, profiles, bearings, tools and machined mechanical components", 0.70),
    Category("FURN", "Furniture",
             "Office and laboratory furniture such as desks, chairs, shelves, cabinets and benches", 0.38),
    Category("PAPER", "Paper and office supplies",
             "Paper, printing, stationery, toner, envelopes and general office consumables", 0.52),
    Category("AIR", "Air travel",
             "Passenger air transport, flight tickets and airline fees", 1.10),
    Category("RAIL", "Rail and public transport",
             "Train tickets, rail passes and public transport travel", 0.04),
    Category("HOTEL", "Accommodation",
             "Hotel stays, overnight accommodation and lodging during business travel", 0.20),
    Category("FOOD", "Catering and food",
             "Catering services, meals, coffee, beverages and food for events and meetings", 0.75),
    Category("ENERGY", "Electricity and heating",
             "Purchased electricity, district heating, natural gas and fuel for buildings", 0.25),
    Category("CONSTR", "Construction and building maintenance",
             "Building works, renovation, installation, repair and maintenance of facilities", 0.33),
    Category("SERV", "Professional services",
             "Consulting, legal, translation, training courses, conference fees and other professional services", 0.10),
)


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in CATALOG])


def catalog_documents() -> list[str]:
    return [f"{c.name}. {c.description}" for c in CATALOG]


def factor_lookup() -> dict[str, float]:
    return {c.code: c.kg_co2e_per_chf for c in CATALOG}
