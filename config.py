"""All constants for the Manufacturing Process Recommender."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "synthetic_sessions.csv"
ARTIFACT_DIR = BASE_DIR / "ml_artifacts"
MODEL_PATH = ARTIFACT_DIR / "process_model.joblib"
LOG_DIR = BASE_DIR / "logs"
SESSIONS_LOG = LOG_DIR / "sessions.csv"
FEEDBACK_LOG = LOG_DIR / "feedback.csv"

RANDOM_STATE = 42
MODEL_FOR_APP = "random_forest"

FEATURE_COLUMNS = [
    "material_type", "part_geometry", "part_size", "strength_level",
    "surface_finish", "tolerance_level", "design_status", "batch_size_cat",
    "urgency_level", "budget_per_part",
]
CATEGORICAL_COLUMNS = ["material_type", "part_geometry"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
TARGET = "recommended_process"
RULE_TARGET = "rule_label"

MATERIALS = ["Plastic", "Rubber", "Aluminum", "Steel", "Titanium"]
METALS = ["Aluminum", "Steel", "Titanium"]

# field -> (question, helper text, {stored value: label shown})
FIELDS = {
    "material_type": ("Primary material", "The main material the part is made of.",
        {m: m for m in MATERIALS}),
    "part_geometry": ("Which best describes the part?", "Pick the closest shape.", {
        1: "Flat plate or cut-out",
        2: "Bent or folded sheet (bracket, enclosure)",
        3: "Solid and chunky (block, round bar)",
        4: "Complex 3D shape (curves, many features)",
        5: "Hollow, or has internal channels"}),
    "part_size": ("Largest dimension of the part", "Measure the longest side.",
        {1: "Under 10 cm", 2: "10-30 cm", 3: "Over 30 cm"}),
    "strength_level": ("Strength needed", "How much load or abuse the part must take.",
        {1: "Visual only", 2: "Functional", 3: "Structural / load-bearing",
         4: "Critical (safety, fatigue or impact loads)"}),
    "surface_finish": ("Surface finish needed", "How the surface should look and feel.",
        {1: "Rough is fine", 2: "Smooth", 3: "Polished"}),
    "tolerance_level": ("Dimensional accuracy needed", "How closely the part must match its drawing.",
        {1: "Loose", 2: "Standard", 3: "High precision"}),
    "design_status": ("Is the design final?", "Choose No if it may still change.",
        {1: "No, prototype or still changing", 2: "Yes, final and ready for production"}),
    "batch_size_cat": ("How many parts do you need?", "Total quantity for this order.",
        {1: "1-10", 2: "11-100", 3: "101-1,000", 4: "1,001-10,000", 5: "More than 10,000"}),
    "urgency_level": ("How soon do you need them?", "Time from order to delivery.",
        {1: "Under 3 days", 2: "3-7 days", 3: "1-3 weeks", 4: "More than 3 weeks"}),
    "budget_per_part": ("Target cost per part (₹)", "What you would like to pay for each part.",
        {1: "Under ₹500", 2: "₹500-2,000", 3: "₹2,000-10,000", 4: "Over ₹10,000"}),
}
STEPS = [
    ("About the part", ["material_type", "part_geometry", "part_size"]),
    ("Requirements", ["strength_level", "surface_finish", "tolerance_level"]),
    ("Production plan", ["design_status", "batch_size_cat", "urgency_level", "budget_per_part"]),
]

CLASSES = {
    "FDM_3D_Printing": ("FDM 3D printing", "Lays down melted plastic filament layer by layer. Cheap and fast, with visible layer lines."),
    "SLA_3D_Printing": ("SLA 3D printing", "Hardens liquid resin with light. Smooth surfaces and fine detail."),
    "SLS_3D_Printing": ("SLS 3D printing", "Fuses nylon powder with a laser. Strong parts, hollow channels possible, no support structures."),
    "Vacuum_Casting": ("Vacuum casting", "Casts polyurethane into a silicone mould. Good finish for small batches without steel tooling."),
    "Injection_Molding": ("Injection molding", "Injects molten plastic or rubber into a steel mould. Cheapest per part at high volume, but the mould is expensive."),
    "CNC_Machining": ("CNC machining", "Cuts the part out of a solid block. High precision, works for metals and tough plastics."),
    "Sheet_Metal": ("Sheet metal fabrication", "Cuts and bends flat metal sheet into flat or folded parts."),
    "SLM_Metal_3D_Printing": ("Metal 3D printing (SLM)", "Melts metal powder with a laser, layer by layer. Complex and hollow metal parts with no tooling, but a high cost per part."),
    "Investment_Casting": ("Investment casting", "Pours molten metal into a ceramic shell made from a wax pattern. Complex metal shapes with a good finish and moderate tooling cost."),
    "Sand_Casting": ("Sand casting", "Pours molten metal into a sand mould. Cheap tooling, suits large parts with a rough finish."),
    "Die_Casting": ("Die casting", "Forces molten aluminum into a steel die. Very low cost per part at high volume, but the die is expensive."),
    "Forging": ("Forging", "Shapes heated metal by pressing or hammering it. The strongest parts for critical loads, but needs forging dies."),
}
CLASS_LABELS = list(CLASSES)

N_CANDIDATES = 200_000
PER_CLASS = 400
NOISE_FRAC = 0.08
