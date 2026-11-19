"""Check that both installers package the same fairchem code and public features."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement


def test_intel_only_distribution_matches_standard_core():
    packages = Path(__file__).resolve().parents[3] / "packages"

    def metadata(package):
        return tomllib.loads((packages / package / "pyproject.toml").read_text())

    regular = metadata("fairchem-core")
    intel = metadata("fairchem-core-xpu")
    assert (packages / "fairchem-core-xpu/src").resolve() == (
        packages / "fairchem-core/src"
    ).resolve()
    standard_project, intel_project = regular["project"], intel["project"]
    assert standard_project["scripts"] == intel_project["scripts"]
    assert standard_project["name"] == intel_project["name"]
    optional = intel_project["optional-dependencies"].copy()
    nvidia = set(optional.pop("nvidia"))
    assert optional == standard_project["optional-dependencies"]
    standard_deps = {
        d for d in standard_project["dependencies"] if Requirement(d).name != "torch"
    }
    intel_deps = {
        d
        for d in intel_project["dependencies"]
        if Requirement(d).name not in {"torch", "triton-xpu"}
    }
    assert intel_deps.isdisjoint(nvidia)
    assert intel_deps | nvidia == standard_deps
    lock = tomllib.loads((packages / "fairchem-core-xpu/uv.lock").read_text())
    torch = next(pkg for pkg in lock["package"] if pkg["name"] == "torch")
    assert torch["version"] == "2.15.0+xpu"
