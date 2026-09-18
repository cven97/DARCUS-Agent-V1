import os
import inspect
import importlib.util
from functools import wraps

# Registre global des outils
# Format: { "tool_name": { "func": function, "desc": description, "args": arg_spec } }
_TOOL_REGISTRY = {}

def tool(name, description):
    """
    Décorateur pour enregistrer une fonction comme outil de l'agent.
    """
    def decorator(func):
        # Obtenir la signature pour inspecter les arguments
        sig = inspect.signature(func)
        args_spec = {}
        for param_name, param in sig.parameters.items():
            # Ne pas inclure self ou context si besoin
            args_spec[param_name] = {
                "default": param.default if param.default is not inspect.Parameter.empty else None,
                "required": param.default is inspect.Parameter.empty
            }
        
        _TOOL_REGISTRY[name] = {
            "func": func,
            "desc": description,
            "args": args_spec
        }
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_registered_tools():
    return _TOOL_REGISTRY

def get_tools_instruction():
    """
    Génère la description textuelle des outils pour le prompt système.
    """
    if not _TOOL_REGISTRY:
        return "Aucun outil disponible."
        
    lines = []
    lines.append("OUTILS DISPONIBLES :")
    for name, info in _TOOL_REGISTRY.items():
        arg_lines = []
        for arg_name, arg_info in info["args"].items():
            req_str = "requis" if arg_info["required"] else "optionnel"
            default_str = f" (défaut: {arg_info['default']})" if not arg_info["required"] and arg_info["default"] is not None else ""
            arg_lines.append(f"    - {arg_name} ({req_str}){default_str}")
            
        args_formatted = "\n".join(arg_lines)
        lines.append(f"- Outil: {name}\n  Description: {info['desc']}\n  Arguments:\n{args_formatted}")
    return "\n\n".join(lines)

def execute_tool(name, arguments):
    """
    Exécute un outil par son nom avec les arguments fournis.
    """
    if name not in _TOOL_REGISTRY:
        return f"Erreur : Outil '{name}' inconnu."
        
    tool_info = _TOOL_REGISTRY[name]
    func = tool_info["func"]
    
    # Valider les arguments requis
    cleaned_args = {}
    for arg_name, arg_info in tool_info["args"].items():
        if arg_name in arguments:
            cleaned_args[arg_name] = arguments[arg_name]
        elif arg_info["required"]:
            return f"Erreur : Argument '{arg_name}' manquant pour l'outil '{name}'."
        else:
            cleaned_args[arg_name] = arg_info["default"]
            
    try:
        return func(**cleaned_args)
    except Exception as e:
        return f"Erreur lors de l'exécution de l'outil '{name}' : {str(e)}"

def load_custom_tools(directory_path):
    """
    Charge dynamiquement les fichiers Python situés dans directory_path 
    et enregistre leurs outils décorés par @tool.
    """
    dir_path = os.path.abspath(directory_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return
        
    for filename in os.listdir(dir_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            file_path = os.path.join(dir_path, filename)
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"[Tools Loader] Impossible de charger {filename} : {e}")
