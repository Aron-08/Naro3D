"""
ui_thread.py — Helper mínimo para tocar widgets de tkinter desde un hilo
que NO es el que corrió root.mainloop().

Por qué hace falta:
    Tkinter no es thread-safe. Este proyecto lanza un threading.Thread(daemon=True)
    cada vez que hay que esperar al modelo (Ollama) para no trabar la ventana
    mientras responde — pero llamar directo a un método de un widget (.config(),
    .insert(), etc.) desde ESE hilo de fondo es undefined behavior. Anda "la
    mayoría de las veces" porque la GIL de CPython serializa las llamadas
    individuales, pero no hay ninguna garantía real de que Tcl/Tk no se cuelgue
    o corrompa el estado interno de la UI — típicamente aparece justo cuando dos
    hilos de fondo actualizan la UI casi al mismo tiempo (dos objetos
    generándose en paralelo), de forma no reproducible.

La solución estándar de tkinter: el hilo de fondo nunca toca el widget
directo. En vez de eso, usa `root.after(0, ...)` para pedirle al hilo que sí
corrió mainloop() que haga la actualización por él — `after()` es seguro de
llamar desde cualquier hilo, y lo que le pasás corre en el próximo ciclo del
event loop de Tk, ya en el hilo correcto.

Uso (desde CUALQUIER hilo, típicamente el que espera la respuesta del modelo):

    from ui_thread import en_hilo_ui

    en_hilo_ui(root, label_estado.config, text="Listo.")
    en_hilo_ui(root, lista_box.insert, tk.END, nombre)
"""


def en_hilo_ui(root, fn, *args, **kwargs) -> None:
    """Programa `fn(*args, **kwargs)` para correr en el hilo de la UI de `root`.

    `root` es la ventana Tk/Toplevel del panel — alcanza con que tenga
    `.after` (todo widget de tkinter lo tiene), así que también sirve pasar
    cualquier widget hijo si no se tiene a mano la referencia al root.

    No hace falta esperar el resultado: las actualizaciones de UI de este
    proyecto (label de estado, insertar en una lista, etc.) son "fire and
    forget" — nadie necesita el valor de vuelta de `fn`.
    """
    root.after(0, lambda: fn(*args, **kwargs))
