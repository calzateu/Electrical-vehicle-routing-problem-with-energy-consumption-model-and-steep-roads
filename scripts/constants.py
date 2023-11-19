
constants_dict = {
    "v_i": 0,
    "v_f": 0,
    "C_r": 0.0064,
    "C_d": 0.7,
    "A": 8.0,
    "rho": 1.2,
    "g": 9.81,
    "v_ab": 6,  # Desired speed per segment
    "m": 1000,
    "acceleration": 0.8,  # From the paper
    "deceleration": -0.9,  # From the paper
    "eta_acceleration_positive": 0.8,
    "eta_acceleration_negative": 1.9,
    "eta_constant_positive": 0.8,
    "eta_constant_negative": 1.9,
    "eta_deceleration_positive": 0.8,
    "eta_deceleration_negative": 1.9,
}

constants_dict['R'] = 0.5 * constants_dict["C_d"] * constants_dict["A"] * constants_dict["rho"]

constants_dict['d_acceleration'] = (constants_dict["v_ab"] ** 2 -
                                    constants_dict["v_i"] ** 2) / (2 * constants_dict["acceleration"])
constants_dict['d_deceleration'] = (constants_dict["v_f"] ** 2 -
                                    constants_dict["v_ab"] ** 2) / (2 * constants_dict["deceleration"])

constants_dict['k'] = constants_dict['d_acceleration'] / constants_dict['d_deceleration']
