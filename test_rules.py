from rules import label_row

DEFAULTS = dict(material_type="Plastic", part_geometry=3, part_size=1, strength_level=1,
                surface_finish=1, tolerance_level=1, design_status=1, batch_size_cat=1,
                urgency_level=1, budget_per_part=1)

CASES = {
    "A": ({}, "FDM_3D_Printing"),
    "B": (dict(material_type="Steel", part_geometry=1), "Sheet_Metal"),
    "C": (dict(material_type="Aluminum", part_geometry=5, budget_per_part=3), "SLM_Metal_3D_Printing"),
    "D": (dict(material_type="Aluminum", part_size=2, batch_size_cat=4, design_status=2,
               strength_level=2, surface_finish=2, urgency_level=3), "Die_Casting"),
    "E": (dict(material_type="Steel", part_size=3, batch_size_cat=2, design_status=2,
               strength_level=2, urgency_level=3), "Sand_Casting"),
    "F": (dict(material_type="Steel", part_size=2, batch_size_cat=3, design_status=2,
               strength_level=4, urgency_level=3), "Forging"),
    "G": (dict(material_type="Steel", part_geometry=4, batch_size_cat=2, design_status=2,
               strength_level=2, surface_finish=2, tolerance_level=2, urgency_level=3), "Investment_Casting"),
    "H": (dict(part_size=2, batch_size_cat=5, design_status=2), "Injection_Molding"),
    "I": (dict(material_type="Rubber", batch_size_cat=2, design_status=2, strength_level=2,
               surface_finish=2, urgency_level=2), "Vacuum_Casting"),
    "J": (dict(surface_finish=2, tolerance_level=2), "SLA_3D_Printing"),
    "K": (dict(part_geometry=5, strength_level=3), "SLS_3D_Printing"),
    "L": (dict(material_type="Steel"), "CNC_Machining"),
}


def test_examples():
    for key, (diff, expected) in CASES.items():
        assert label_row({**DEFAULTS, **diff}) == expected, key


if __name__ == "__main__":
    test_examples()
    print("All 12 example rows pass.")
