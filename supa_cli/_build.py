"""Keep configured exclusions out of Hatchling's forced sdist files."""

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class SdistBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Hatchling force-includes parent VCS ignore files outside the include list.
        build_data["force_include"] = {
            source: target
            for source, target in build_data["force_include"].items()
            if not self.build_config.path_is_excluded(target)
        }
