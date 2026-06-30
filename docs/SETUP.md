# Setup — entornos virtuales separados por framework

TensorFlow y PyTorch fijan versiones en conflicto de `numpy`, `protobuf`, etc.
**Nunca** instales ambos en el mismo entorno. Usa tres venvs aislados, todos
sobre **Python 3.11** (mejor compatibilidad TF + PyTorch + export móvil en Windows).

El repo trae un script que crea los tres: `setup_envs.ps1`.

```powershell
# desde la raíz del repo
./setup_envs.ps1
```

Crea:

| venv               | para           | requirements                  |
|--------------------|----------------|-------------------------------|
| `tools/.venv`      | conversor      | `requirements-tools.txt`      |
| `PyTorch/.venv`    | modelo PyTorch | `requirements-pytorch.txt`    |
| `TensorFlow/.venv` | modelo TF      | `requirements-tensorflow.txt` |

## Activar un venv (PowerShell)

```powershell
tools\.venv\Scripts\Activate.ps1        # conversor
PyTorch\.venv\Scripts\Activate.ps1      # PyTorch
TensorFlow\.venv\Scripts\Activate.ps1   # TensorFlow
deactivate                              # salir
```

Si `Activate.ps1` está bloqueado:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## ¿Por qué Python 3.11?
- `tensorflow` 2.15/2.16 tiene wheels para CPython 3.9-3.11 en Windows; 3.12/3.13
  son inestables.
- `torch==2.2.2` (con el export lite-interpreter móvil) tiene wheels limpias en 3.11.
- El Python de la **Microsoft Store** está aislado y rompe los venvs — usa el
  instalador de python.org 3.11.

Instalar Python 3.11 (si no está):
```powershell
winget install --id Python.Python.3.11 -e --scope user
```
