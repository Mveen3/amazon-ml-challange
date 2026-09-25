"""Country parsing profiles (hand-written abbreviation knowledge, no external data).

Country is only used to pick parsing rules and threshold sets; it is never a
model feature. Unknown countries fall back to ``generic`` (the task treats the
country set as open).
"""
from __future__ import annotations

from dataclasses import dataclass, field

_COUNTRY_TO_PROFILE = {
    "us": "us", "usa": "us", "united states": "us", "united states of america": "us",
    "india": "india", "in": "india", "bharat": "india",
    "france": "france", "fr": "france",
}


def profile_name(country: str | None) -> str:
    return _COUNTRY_TO_PROFILE.get((country or "").strip().casefold(), "generic")


# ----------------------------------------------------------------------------- legal forms
# Each entry: token tuple (after lower-casing, punctuation removal and single-letter
# collapsing, so "S.A.S." -> "sas", "L.L.C." -> "llc") -> canonical class.
_LEGAL_EN = {
    ("limited", "liability", "company"): "llc", ("limited", "liability", "partnership"): "llp",
    ("limited", "partnership"): "lp", ("llc",): "llc", ("pllc",): "pllc", ("llp",): "llp", ("lp",): "lp",
    ("lllp",): "llp", ("plc",): "plc", ("inc",): "inc", ("lnc",): "inc", ("incorporated",): "inc", ("corp",): "corp",
    ("corporation",): "corp", ("co",): "co", ("company",): "co", ("ltd",): "ltd", ("limited",): "ltd",
}
_LEGAL_IN = {
    ("private", "limited"): "pvt_ltd", ("pvt", "ltd"): "pvt_ltd", ("pvt", "limited"): "pvt_ltd",
    ("private", "ltd"): "pvt_ltd", ("p", "ltd"): "pvt_ltd", ("pvtltd",): "pvt_ltd", ("pvt",): "pvt_ltd",
    ("public", "limited"): "public_ltd", ("opc",): "opc", ("one", "person", "company"): "opc",
}
_LEGAL_FR = {
    ("sarl",): "sarl", ("sas",): "sas", ("sasu",): "sasu", ("eurl",): "eurl", ("sa",): "sa", ("sci",): "sci",
    ("snc",): "snc", ("scop",): "scop", ("ei",): "ei", ("eirl",): "eirl", ("selarl",): "selarl",
    ("scm",): "scm", ("gie",): "gie", ("sca",): "sca", ("scs",): "scs",
}
LEGAL_FAMILY = {
    "pvt_ltd": "ltd", "ltd": "ltd", "public_ltd": "ltd", "opc": "ltd", "plc": "ltd",
    "inc": "inc", "corp": "inc", "co": "inc", "llc": "llc", "pllc": "llc", "llp": "partner", "lp": "partner",
    "sarl": "fr_sarl", "eurl": "fr_sarl", "selarl": "fr_sarl", "sas": "fr_sas", "sasu": "fr_sas",
    "sa": "fr_sa", "sci": "fr_civil", "scm": "fr_civil", "snc": "fr_other", "scop": "fr_other",
    "ei": "fr_other", "eirl": "fr_other", "gie": "fr_other", "sca": "fr_other", "scs": "fr_other",
}

# ----------------------------------------------------------------------------- abbreviations
_ADDR_US = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "ter": "terrace", "terr": "terrace",
    "trl": "trail", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "sq": "square", "pt": "point",
    "mt": "mount", "ft": "fort", "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "cv": "cove", "xing": "crossing", "expy": "expressway", "fwy": "freeway",
    "jct": "junction", "mtn": "mountain", "rdg": "ridge", "vly": "valley", "vw": "view", "hts": "heights",
    "holw": "hollow", "lk": "lake", "crk": "creek", "spgs": "springs", "sta": "station", "twp": "township",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "po": "post office", "rte": "route", "rt": "route", "ext": "extension",
}
_ADDR_IN = {
    "rd": "road", "st": "street", "str": "street", "bldg": "building", "apt": "apartment", "apts": "apartments",
    "soc": "society", "ngr": "nagar", "sec": "sector", "fl": "floor", "flr": "floor", "blk": "block",
    "dist": "district", "vill": "village", "vil": "village", "tal": "taluk", "po": "post office",
    "mkt": "market", "stn": "station", "hosp": "hospital", "govt": "government", "opp": "opposite",
    "nr": "near", "cmplx": "complex", "extn": "extension", "ext": "extension", "indl": "industrial",
    "ind": "industrial", "estt": "estate", "chowk": "chowk", "mg": "mahatma gandhi",
}
_ADDR_FR = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
    "pl": "place", "rte": "route", "ch": "chemin", "chem": "chemin", "imp": "impasse", "all": "allee",
    "fbg": "faubourg", "sq": "square", "crs": "cours", "qu": "quai", "pas": "passage", "res": "residence",
    "lot": "lotissement", "st": "saint", "ste": "sainte", "bat": "batiment", "esc": "escalier",
    "apt": "appartement", "gal": "general", "mal": "marechal", "pdt": "president", "prof": "professeur",
    "dr": "docteur", "za": "zone artisanale", "zi": "zone industrielle", "zac": "zone amenagement",
    "cite": "cite", "hlm": "hlm", "rpt": "rond point", "sen": "sente", "vla": "villa",
}
_NAME_ABBR_COMMON = {
    "intl": "international", "int'l": "international", "mfg": "manufacturing", "svcs": "services",
    "svc": "service", "bros": "brothers", "assoc": "associates", "assocs": "associates", "natl": "national",
    "mgmt": "management", "mgt": "management", "tech": "tech", "engg": "engineering", "eng": "engineering",
    "ents": "enterprises", "hosp": "hospital", "univ": "university",
    "dept": "department", "grp": "group", "sys": "systems", "solns": "solutions",
    "trdg": "trading", "inds": "industries", "ind": "industries", "exp": "exports", "imp": "imports",
}
_NAME_ABBR_FR = {"cie": "compagnie", "ets": "etablissements", "ste": "societe", "assoc": "association",
                 "st": "saint"}

_ADDR_PREFIX = {"no", "hno", "dno", "hn", "door", "dhor", "house", "plot", "flat", "shop", "khasra",
                "survey", "sy", "sno", "gat"}
_ADDR_PREFIX_HEAD = {"h", "d", "s", "door", "dhor", "house", "plot", "flat", "shop", "khasra", "kh", "survey", "sy"}
_LANDMARK = {"near", "nr", "opp", "opposite", "behind", "beside", "besides", "adjacent", "adj", "adjoining",
             "facing", "pres", "face"}
_PHONE = {"ph", "phone", "mob", "mobile", "tel", "telephone", "contact", "cell", "fax", "mo"}
_HONORIFIC = {"ms", "messrs", "mr", "mrs"}
_STOP_EN = {"the", "of", "and", "a", "an", "for", "&"}
_STOP_FR = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "au", "aux", "en", "the", "and", "of"}


@dataclass
class Profile:
    name: str
    amp: str = "and"
    legal: dict = field(default_factory=dict)
    addr_abbr: dict = field(default_factory=dict)
    name_abbr: dict = field(default_factory=dict)
    postcode_len: int | None = None
    postcode_always: bool = False  # India: any standalone 6-digit token is a PIN
    number_suffix: tuple = ("bis", "ter", "quater")
    letter_bis: bool = False  # France: "9 B" means 9 bis
    stop: set = field(default_factory=lambda: set(_STOP_EN))
    addr_prefix: set = field(default_factory=lambda: set(_ADDR_PREFIX))
    addr_prefix_head: set = field(default_factory=lambda: set(_ADDR_PREFIX_HEAD))
    landmark: set = field(default_factory=lambda: set(_LANDMARK))
    phone: set = field(default_factory=lambda: set(_PHONE))
    honorific: set = field(default_factory=lambda: set(_HONORIFIC))

    def __post_init__(self):
        self.legal_max = max((len(k) for k in self.legal), default=1)


BASE_PROFILES = {
    "us": Profile("us", legal={**_LEGAL_EN, **_LEGAL_IN}, addr_abbr=_ADDR_US, name_abbr=_NAME_ABBR_COMMON,
                  postcode_len=5),
    "india": Profile("india", legal={**_LEGAL_EN, **_LEGAL_IN}, addr_abbr=_ADDR_IN, name_abbr=_NAME_ABBR_COMMON,
                     postcode_len=6, postcode_always=True),
    "france": Profile("france", amp="et", legal={**_LEGAL_EN, **_LEGAL_FR},
                      addr_abbr=_ADDR_FR, name_abbr={**_NAME_ABBR_COMMON, **_NAME_ABBR_FR}, postcode_len=5,
                      letter_bis=True, stop=set(_STOP_FR)),
    "generic": Profile("generic", legal={**_LEGAL_EN, **_LEGAL_IN, **_LEGAL_FR}, addr_abbr={},
                       name_abbr=_NAME_ABBR_COMMON, postcode_len=None),
}

_RUNTIME_CACHE: dict = {}


def runtime_profile(name: str, tables=None) -> Profile:
    """Profile with mined abbreviation maps merged in (hand-written rules win)."""
    key = (name, id(tables))
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]
    base = BASE_PROFILES.get(name, BASE_PROFILES["generic"])
    prof = Profile(**{k: v for k, v in base.__dict__.items() if k != "legal_max"})
    if tables is not None:
        prof.addr_abbr = {**tables.addr_abbr.get(name, {}), **base.addr_abbr}
        prof.name_abbr = {**tables.name_abbr.get(name, {}), **base.name_abbr}
    _RUNTIME_CACHE[key] = prof
    return prof


ALL_LEGAL_TOKENS = {t for p in BASE_PROFILES.values() for k in p.legal for t in k}
