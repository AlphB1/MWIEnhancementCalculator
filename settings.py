from constant import *


class Setting:
    def __init__(self, fp='setting.json'):
        import json
        j = json.load(open(fp, 'r'))
        drink_concentration = 1 + 0.1 * (1 + BONUS[j["guzzling_pouch_level"]]) if j["guzzling_pouch"] else 1.0

        skill_level = j["skill_level"] + drink_concentration * (
                (3 if j["tea"]['enhancing tea'] else 0) +
                (6 if j["tea"]['super enhancing tea'] else 0) +
                (8 if j["tea"]['ultra enhancing tea'] else 0)
        )
        recommended_level = j["recommended_level"]
        level_rate = ((skill_level - recommended_level) * 0.0005) \
            if (skill_level >= recommended_level) \
            else (-0.5 * (1 - skill_level / recommended_level))
        enhancer_buff = TOOLS_LEVEL[j["enhancer_type"]] * (1 + BONUS[j["enhancer_level"]])
        enhancer_buff = round(enhancer_buff, 4)  # doh-nuts website has this rounding, and I don't know why
        self.blessed_rate = 0.01 * drink_concentration if j["tea"]['blessed tea'] else 0.0
        self.enhance_rate_mod = 1 + level_rate + 0.0005 * j["laboratory_level"] + enhancer_buff
        self.target_level = j["target_level"]

    def enhance_success_rate(self, enhance_level: int):
        return BASE_SUCCESS_RATE[enhance_level] * self.enhance_rate_mod
