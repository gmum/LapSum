from pathlib import Path
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


here = Path(__file__).parent
root = here / "lapsum"
ext_modules = []

def get_pytorch_version_tuple():
    return tuple(map(int, torch.__version__.split('+')[0].split('.')[:3]))


def replace_in_file(filepath: Path, old: str, new: str):
    text = filepath.read_text(encoding='utf-8')
    if old not in text:
        return  # no changes needed
    new_text = text.replace(old, new)
    filepath.write_text(new_text, encoding='utf-8')
    print(f"Updated: {filepath}")


def rel(p: Path) -> str:
    return str(p.relative_to(here)).replace("\\", "/")  # for Windows


def collect_extensions(module: str):
    base = root / module
    cpu = base / "cpu"
    cuda = base / "cuda"

    for src in cpu.glob("*.cpp"):
        ext_modules.append(
            CppExtension(
                name=f"lapsum.{module}.extension.{src.stem}_cpu",
                sources=[rel(src)],
                extra_link_args=["-Wl,--no-as-needed", "-lm", "-fopenmp"],
            )
        )

    cuda_pairs = {
        cu.stem: (cu, base / "cuda" / f"{cu.stem}_cuda.cpp")
        for cu in cuda.glob("*.cu")
    }

    for name, (cu, cpp) in cuda_pairs.items():
        ext_modules.append(
            CUDAExtension(
                name=f"lapsum.{module}.extension.{name}_cuda",
                sources=[rel(cu), rel(cpp)],
                extra_compile_args={
                    "cxx": ["-O2"],
                    "nvcc": ["-O2", "--expt-relaxed-constexpr"]
                },
            )
        )

# packages_selection = ["topk", "rank", "sort", "permute"]
packages_selection = ["topk"]

for sub in packages_selection:
    pytorch_version = get_pytorch_version_tuple()
    version_geq_2_6 = pytorch_version >= (2, 6, 0)

    print(f"Detected PyTorch version: {torch.__version__}")
    if version_geq_2_6:
        print("PyTorch >= 2.6, replacing '.type()' with '.scalar_type()'...")
        old, new = '.type()', '.scalar_type()'
    else:
        print("PyTorch < 2.6, replacing '.scalar_type()' with '.type()'...")
        old, new = '.scalar_type()', '.type()'

    for filepath in (root / sub).rglob('*'):
        if filepath.suffix in ('.cpp', '.cu') and filepath.is_file():
            replace_in_file(filepath, old, new)

    collect_extensions(sub)

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
