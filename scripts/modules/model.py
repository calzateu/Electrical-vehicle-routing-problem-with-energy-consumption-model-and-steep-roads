import math


# Energy Consumption Model Definition

# Definition of efficiency functions
# IMPORTANT: They use regression analysis to estimate each function. That, with data
# simulated by the Global Simulation Platform (GSP) of Volvo Group
# The values or functions we need to define are further below, in the last parameters.


def eta_acceleration(mechanical_energy, information):
    if mechanical_energy >= 0:
        return information['eta_acceleration_positive']
    else:
        return information['eta_acceleration_negative']


def eta_constant(mechanical_energy, information):
    if mechanical_energy >= 0:
        return information['eta_constant_positive']
    else:
        return information['eta_constant_negative']


def eta_deceleration(mechanical_energy, information):
    if mechanical_energy >= 0:
        return information['eta_deceleration_positive']
    else:
        return information['eta_deceleration_negative']


# ## Model Equations
# Refer to section 3.4 of the base article.

def alpha_acceleration(angle, information, distance_acceleration=None):
    if distance_acceleration is not None:
        d_a = distance_acceleration
    else:
        d_a = information['d_acceleration']
    acc = information['acceleration']
    g = information['g']
    C_r = information['C_r']

    alph_acceleration = (d_a * (
            acc + g * math.sin((angle * math.pi) / 180) + g * C_r * math.cos((angle * math.pi) / 180))) / 3600

    return alph_acceleration


def alpha_constant(angle, distance, information, distance_acceleration=None, distance_deceleration=None):
    g = information['g']
    d_ab = distance
    if distance_acceleration is not None:
        d_a = distance_acceleration
    else:
        d_a = information['d_acceleration']
    if distance_deceleration is not None:
        d_d = distance_deceleration
    else:
        d_d = information['d_deceleration']
    C_r = information['C_r']

    alph_constant = (g * (d_ab - d_a - d_d) * (
            math.sin((angle * math.pi) / 180) + C_r * math.cos((angle * math.pi) / 180))) / 3600

    return alph_constant


def alpha_deceleration(angle, information, distance_deceleration=None):
    if distance_deceleration is not None:
        d_d = distance_deceleration
    else:
        d_d = information['d_deceleration']
    decc = information['deceleration']
    g = information['g']
    C_r = information['C_r']

    alph_deceleration = (d_d * (
            decc + g * math.sin((angle * math.pi) / 180) + g * C_r * math.cos((angle * math.pi) / 180))) / 3600

    return alph_deceleration


def beta_acceleration(information, distance_acceleration=None, target_speed=None):
    R = information['R']
    if distance_acceleration is not None:
        d_a = distance_acceleration
    else:
        d_a = information['d_acceleration']
    v_i = information['v_i']

    if target_speed is not None:
        v_ab = target_speed
    else:
        v_ab = information['v_ab']

    beth_acceleration = (R * d_a * (v_i ** 2 + (v_ab ** 2 - v_i ** 2) / 2)) / 3600

    return beth_acceleration


def beta_constant(distance, information, distance_acceleration=None, distance_deceleration=None,
                  target_speed=None):
    R = information['R']
    d_ab = distance
    if distance_acceleration is not None:
        d_a = distance_acceleration
    else:
        d_a = information['d_acceleration']
    if distance_deceleration is not None:
        d_d = distance_deceleration
    else:
        d_d = information['d_deceleration']

    if target_speed is not None:
        v_ab = target_speed
    else:
        v_ab = information['v_ab']

    beth_constant = (R * (d_ab - d_a - d_d) * v_ab ** 2) / 3600

    return beth_constant


def beta_deceleration(information, distance_deceleration=None, target_speed=None):
    R = information['R']
    if distance_deceleration is not None:
        d_d = distance_deceleration
    else:
        d_d = information['d_deceleration']

    if target_speed is not None:
        v_ab = target_speed
    else:
        v_ab = information['v_ab']

    v_f = information['v_f']

    beth_deceleration = (R * d_d * (v_ab ** 2 + (v_f ** 2 - v_ab ** 2) / 2)) / 3600

    return beth_deceleration


# def alpha_a_b(a, b, angles, information,):
#     alpha = alpha_acceleration(a, b, angles, information) + alpha_constant(a, b,
#                     angles, information) + alpha_deceleration(a, b, angles, information)
#     return alpha


# def beta_a_b(a, b, angles, information):
#     beta = beta_acceleration(a, b, angles, information) + beta_constant(a, b,
#                     angles, information) + beta_deceleration(a, b, angles, information)
#     return beta


def acceleration_func(angle, information, distance_acceleration=None, target_speed=None):
    alpha = alpha_acceleration(angle, information, distance_acceleration)  # Estimation without energy efficiency
    beta = beta_acceleration(information, distance_acceleration, target_speed)  # Estimation without energy efficiency
    mechanical_energy = alpha * information['m'] + beta
    eta = eta_acceleration(mechanical_energy, information)
    energy_acceleration = mechanical_energy / eta
    return energy_acceleration


def constant_func(angle, distance, information, distance_acceleration=None, distance_deceleration=None,
                  target_speed=None):
    alpha = alpha_constant(angle, distance, information, distance_acceleration,
                           distance_deceleration)  # Estimation without energy efficiency
    beta = beta_constant(distance, information, distance_acceleration, distance_deceleration,
                         target_speed)  # Estimation without energy efficiency
    mechanical_energy = alpha * information['m'] + beta
    eta = eta_constant(mechanical_energy, information)
    energy_constant = mechanical_energy / eta
    return energy_constant


def deceleration_func(angle, information, distance_deceleration=None, target_speed=None):
    alpha = alpha_deceleration(angle, information, distance_deceleration)  # Energy estimation without efficiency
    beta = beta_deceleration(information, distance_deceleration, target_speed)  # Energy estimation without efficiency
    mechanical_energy = alpha * information['m'] + beta
    eta = eta_deceleration(mechanical_energy, information)
    energy_deceleration = mechanical_energy / eta
    return energy_deceleration


def verify_distances(distance, information):
    if information['d_acceleration'] + information['d_deceleration'] > distance:
        distance_deceleration = distance / (1 + information['k'])
        distance_acceleration = information['k'] * distance_deceleration

        target_speed = math.sqrt(information['v_i'] ** 2 + 2 * information['acceleration'] * distance_acceleration)

        return distance_acceleration, distance_deceleration, target_speed, True

    return None, None, None, False


def energy_between_a_b(angle, distance, information):
    # energy = alpha_a_b(a, b, angles, information)*information['m'] + beta_a_b(a, b,
    #                angles, information)

    distance_acceleration, distance_deceleration, target_speed, recalculated_distances = verify_distances(
        distance, information
    )

    energy = acceleration_func(
        angle, information, distance_acceleration=distance_acceleration, target_speed=target_speed
    ) + constant_func(
        angle, distance, information, distance_acceleration=distance_acceleration,
        distance_deceleration=distance_deceleration, target_speed=target_speed
    ) + deceleration_func(
        angle, information, distance_deceleration=distance_deceleration, target_speed=target_speed
    )

    return energy


def energy_distance_and_energy_per_meter(a, b, angles, distances, information):
    angle = angles[(a, b)]
    distance = distances[(a, b)]

    energy = energy_between_a_b(angle, distance, information)

    energy_per_meter = energy / distance

    return angle, distance, energy, energy_per_meter


# ## Initialize Parameters
def initialize_information(v_i, v_f, v_ab, acceleration, deceleration, m, C_r, C_d,
                           A, rho, g, eta_acceleration_positive, eta_acceleration_negative,
                           eta_constant_positive, eta_constant_negative, eta_deceleration_positive,
                           eta_deceleration_negative):
    information = dict()

    information['v_i'] = v_i
    information['v_f'] = v_f
    information['v_ab'] = v_ab
    information['acceleration'] = acceleration
    information['deceleration'] = deceleration
    information['m'] = m

    information['C_r'] = C_r
    information['C_d'] = C_d
    information['A'] = A
    information['rho'] = rho
    information['g'] = g

    information['R'] = 0.5 * C_d * A * rho
    information['d_acceleration'] = (v_ab ** 2 - v_i ** 2) / (2 * acceleration)
    information['d_deceleration'] = (v_f ** 2 - v_ab ** 2) / (2 * deceleration)

    information['k'] = information['d_acceleration'] / information['d_deceleration']

    information['eta_acceleration_positive'] = eta_acceleration_positive
    information['eta_acceleration_negative'] = eta_acceleration_negative
    information['eta_constant_positive'] = eta_constant_positive
    information['eta_constant_negative'] = eta_constant_negative
    information['eta_deceleration_positive'] = eta_deceleration_positive
    information['eta_deceleration_negative'] = eta_deceleration_negative

    return information
