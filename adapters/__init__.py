# adapters/__init__.py
#
# makes 'adapters' a Python package and provides load_adapter(site):
# - reads site["adapter"] from sites-advanced.yaml
# - imports the matching module from this folder 'adapters'
#
# steps to add a new adapter:
# - create adapters/[adapter_name].py
# - implement function 'fetch(url)' in each module which returns 'list[pd.DataFrame]'
# - set adapter like "adapter_name" in sites-advanced.yaml

import importlib
import os
from pathlib import Path

sites_yaml = 'sites-advanced.yaml'

# scan 'adapters/' folder and return all module names except __init__
def _installed_adapters() -> list[str]:    
    adapters_dir = Path(__file__).parent

    # stem = filename without .py
    return sorted([file.stem for file in adapters_dir.glob("*.py") if file.stem != "__init__"])
    
# load and return the adapter module, so that 'adapter.fetch(url)' can be called
def load_adapter(site: dict):
    adapter_name = site.get("adapter")
    available = _installed_adapters()
    if not adapter_name:
        raise ValueError(
            f"Site '{site['name']}' has no 'adapter' field in '{sites_yaml}'\n"
            f"Add 'adapter \"adapter_name\"' for instance in '{sites_yaml}'\n"
            + "\n".join(f"  adapter: \"{a}\"" for a in available)
        )
        
    try:
        # the following is the same as 'importlib.import_module("adapters.adapter_name")'
        return importlib.import_module(f"adapters.{adapter_name}")
    except ModuleNotFoundError:
        raise ValueError(
            f"Adapter '{adapter_name}' not found under 'adapters/'\n"
            f"Installed adapters in 'adapters/':\n"
            + "\n".join(f"  {a}" for a in available)
            + f"\nCreate 'adapters/{adapter_name}.py' to add to 'adapters/'"
        )
