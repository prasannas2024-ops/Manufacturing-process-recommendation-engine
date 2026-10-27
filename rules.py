"""Labelling rules (single source of truth). First matching rule wins."""
from config import FIELDS, METALS, MATERIALS


def validate(values):
    """Return an error message, or None if the values are valid.

    Values must already be converted (numeric fields as int, material as str).
    """
    for name, (_, _, allowed) in FIELDS.items():
        if name not in values or values[name] not in allowed:
            return f"Please choose a valid answer for: {FIELDS[name][0]}."
    if values["part_geometry"] == 2 and values["material_type"] not in ("Aluminum", "Steel"):
        return ("A bent or folded sheet is only possible in Aluminum or Steel. "
                "Please change the material or the part shape.")
    return None


def label_row(row):
    mat = row["material_type"]
    geo = row["part_geometry"]
    size = row["part_size"]
    st = row["strength_level"]
    fin = row["surface_finish"]
    tol = row["tolerance_level"]
    des = row["design_status"]
    bat = row["batch_size_cat"]
    urg = row["urgency_level"]
    bud = row["budget_per_part"]

    if mat in METALS:
        al_st = mat in ("Aluminum", "Steel")
        proto = des == 1 or bat == 1
        if al_st and geo in (1, 2):
            return "Sheet_Metal"                                        # M1
        if proto and geo >= 4 and bud >= 3:
            return "SLM_Metal_3D_Printing"                              # M2
        if proto:
            return "CNC_Machining"                                      # M3
        if st == 4 and geo in (3, 4) and bat >= 3 and urg >= 3:
            return "Forging"                                            # M4
        if mat == "Aluminum" and bat >= 4 and geo in (3, 4) and size <= 2 and urg >= 3:
            return "Die_Casting"                                        # M5
        if (al_st and geo >= 3 and fin == 1 and urg >= 3
                and (tol == 1 or (size == 3 and tol == 2))):
            return "Sand_Casting"                                       # M6
        if geo >= 4 and size <= 2 and urg >= 3:
            return "Investment_Casting"                                 # M7
        if geo == 5 and bud >= 3:
            return "SLM_Metal_3D_Printing"                              # M8
        if mat == "Steel" and geo == 3 and bat >= 4:
            return "Forging"                                            # M9
        return "CNC_Machining"                                          # M10

    solid = geo != 5
    if des == 2 and bat >= 4 and solid:
        return "Injection_Molding"                                      # P1
    if des == 2 and bat == 3 and solid and urg == 4 and bud == 1:
        return "Injection_Molding"                                      # P2
    if size == 3 and solid and tol == 3:
        return "CNC_Machining"                                          # P3
    if size == 3:
        return "FDM_3D_Printing"                                        # P4
    if des == 2 and bat == 3 and solid and urg >= 2:
        return "Vacuum_Casting"                                         # P5
    if des == 2 and bat == 2 and solid and urg >= 2 and (fin >= 2 or mat == "Rubber"):
        return "Vacuum_Casting"                                         # P6
    if tol == 3 and st >= 2 and solid and bud >= 3:
        return "CNC_Machining"                                          # P7
    if mat == "Rubber" and st >= 2:
        return "SLS_3D_Printing"                                        # P8
    if mat == "Rubber":
        return "FDM_3D_Printing"                                        # P9
    if st >= 3:
        return "SLS_3D_Printing"                                        # P10
    if fin >= 2 or tol == 3:
        return "SLA_3D_Printing"                                        # P11
    if geo == 5:
        return "SLS_3D_Printing"                                        # P12
    return "FDM_3D_Printing"                                            # P13
