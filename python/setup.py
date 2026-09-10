from setuptools import Distribution, setup
from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    """Mark the ctypes payload as platform-specific without a CPython ABI."""

    def has_ext_modules(self) -> bool:
        return True


class WindowsX64Wheel(bdist_wheel):
    """Emit the stable py3-none-win_amd64 tag for the bundled x64 DLLs."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", "win_amd64"


setup(
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": WindowsX64Wheel},
)
