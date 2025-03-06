import platform
import subprocess

import psutil
from ovos_config import Configuration
from ovos_lang_parser import pronounce_lang
from ovos_number_parser import pronounce_number

from ovos_workshop.decorators import intent_handler
from ovos_workshop.skills import OVOSSkill


class SystemDiagnosticsSkill(OVOSSkill):

    @intent_handler("query_kernel_version.intent")
    def handle_get_os(self, message):
        kernel = platform.release().split("-")[0]  # simplify version number
        self.speak_dialog("kernel_version",
                          {"os_info": f"{platform.system()} {kernel}"})

    @intent_handler("query_gpu.intent")
    def handle_get_gpu(self, message):
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = has_nvidia_gpu()
        if gpu:
            self.speak_dialog("has_gpu")
        else:
            self.speak_dialog("no_gpu")

    @intent_handler("query_cpu_usage.intent")
    def handle_get_cpu(self, message):
        self.speak_dialog("cpu_percent",
                          {"cpu": pronounce_number(psutil.cpu_percent(), lang=self.lang)})

    @intent_handler("query_memory_usage.intent")
    def handle_get_memory(self, message):
        m = psutil.virtual_memory()
        memory = pronounce_number(m.percent, lang=self.lang)
        self.speak_dialog("current_memory", {"memory": memory})
        memory = nice_bytes(m.available)
        total = nice_bytes(m.total)
        self.speak_dialog("available_memory",
                          {"memory": memory, "total": total})

    @intent_handler("query_user_location.intent")
    def handle_user_location(self, message):
        location = self.location_pretty
        self.speak_dialog("user_location", {"location": location})

    @intent_handler("query_ovos_location.intent")
    def handle_default_location(self, message):
        loc = Configuration().get("location", {}).get("city", {}).get("name")
        self.speak_dialog("system_location", {"location": loc})

    @intent_handler("query_user_lang.intent")
    def handle_what_lang_am_i_speaking(self, message):
        lang = pronounce_lang(self.lang, self.lang)
        self.speak_dialog("current_lang", {"lang": lang})

    @intent_handler("query_primary_lang.intent")
    def handle_primary_lang(self, message):
        lang = pronounce_lang(self.core_lang, self.lang)
        self.speak_dialog("primary_lang", {"lang": lang})

    @intent_handler("query_extra_langs.intent")
    def handle_secondary_langs(self, message):
        if self.secondary_langs:
            langs = ', '.join([pronounce_lang(l, self.lang) for l in self.secondary_langs])
            self.speak_dialog("secondary_langs", {"langs": langs})
        else:
            self.speak_dialog("no_secondary_langs")

    @intent_handler("query_langs.intent")
    def handle_get_languages(self, message):
        if self.core_lang != self.lang:
            self.handle_what_lang_am_i_speaking(message)
        self.handle_primary_lang(message)
        if self.secondary_langs:
            self.handle_secondary_langs(message)

    ######################
    # TODO - query via bus
    def handle_list_skills(self, message):
        # TODO - placeholder
        skills = [self.skill_id]
        self.speak(f"I have the following skills installed: {', '.join(skills)}")

    def handle_list_tts(self, message):
        # TODO - placeholder
        plugins = []
        self.speak(f"I have the following text to speech plugins installed: {', '.join(plugins)}")

    def handle_list_stt(self, message):
        # TODO - placeholder
        plugins = []
        self.speak(f"I have the following speech to text plugins installed: {', '.join(plugins)}")

    def handle_list_vad(self, message):
        # TODO - placeholder
        plugins = []
        self.speak(f"I have the following voice activity detection plugins installed: {', '.join(plugins)}")


def has_nvidia_gpu():
    try:
        subprocess.check_output('nvidia-smi')
        return True
    except Exception:  # this command not being found can raise quite a few different errors depending on the configuration
        pass
    return False


def nice_bytes(number, speech=True, binary=True, gnu=False):
    """
    turns a number of bytes into a string using appropriate units
    prefixes - https://en.wikipedia.org/wiki/Binary_prefix
    spoken binary units - https://en.wikipedia.org/wiki/Kibibyte
    implementation - http://stackoverflow.com/a/1094933/2444609
    :param number: number of bytes (int)
    :param speech: spoken form (True) or short units (False)
    :param binary: 1 kilobyte = 1024 bytes (True) or 1 kilobyte = 1000 bytes (False)
    :param gnu: say only order of magnitude (bool)  - 100 Kilo (True) or 100 Kilobytes (False)
    :return: nice bytes (str)
    """

    if speech and gnu:
        default_units = ['Bytes', 'Kilo', 'Mega', 'Giga', 'Tera', 'Peta',
                         'Exa', 'Zetta', 'Yotta']
    elif speech and binary:
        default_units = ['Bytes', 'Kibibytes', 'Mebibytes', 'Gibibytes',
                         'Tebibytes', 'Pebibytes', 'Exbibytes', 'Zebibytes',
                         'Yobibytes']
    elif speech:
        default_units = ['Bytes', 'Kilobytes', 'Megabytes', 'Gigabytes',
                         'Terabytes', 'Petabytes', 'Exabytes', 'Zettabytes',
                         'Yottabytes']
    elif gnu:
        default_units = ['B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    elif binary:
        default_units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB',
                         'YiB']
    else:
        default_units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']

    units = default_units

    n = 1024 if binary else 1000

    for unit in units[:-1]:
        if abs(number) < n:
            if number == 1 and speech and not gnu:
                # strip final "s"
                unit = unit[:-1]
            return "%3.1f %s" % (number, unit)
        number /= n
    return "%.1f %s" % (number, units[-1])



