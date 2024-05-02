from tkinter import *
from tkinter import ttk
from tkinter.ttk import *
import typing as tp
from dataclasses import dataclass

@dataclass
class ElementData:
    element: Widget
    variable: Variable | None = None

class TkinterElements:
    def __init__(self, tab_control: ttk.Notebook, tab_name: str) -> None:
        self.tab_control = tab_control
        self.window = ttk.Frame(self.tab_control)
        self.tab_name = tab_name
        self.tab_control.add(self.window, text = tab_name)

    def add_label(self, text: str, grid: tuple[int, int]) -> None:
        lbl = Label(self.window, text=text)
        # lbl.grid(column=grid[0], row=grid[1])
        lbl.pack()
        # self.tab_control.pack()

    def add_box(self, grid: tuple[int, int], label_name: str, values: str, default_value: str) -> ElementData:
        self.add_label(label_name, grid)
        combo = Combobox(self.window, state='readonly')
        combo['values']= values
        combo.current(values.index(default_value))
        # combo.grid(column=grid[0], row=grid[1] + 1)
        combo.pack()
        # self.tab_control.pack()
        return ElementData(combo)

    def add_checker(self, grid: tuple[int, int], label_name: str, button_text: str) -> ElementData:
        self.add_label(label_name, grid)
        checker_var = BooleanVar(value=True)
        checker = Checkbutton(self.window,
                                text=button_text,
                                variable=checker_var)
        # checker.grid(column=grid[0], row=grid[1] + 1)
        checker.pack()
        # self.tab_control.pack()
        return ElementData(checker, checker_var)

    def add_progressbar(self, grid: tuple[int, int]) -> ElementData:
        style = ttk.Style()
        style.theme_use('default')
        style.configure("black.Horizontal.TProgressbar", background='black')
        bar = Progressbar(self.window, length=200, style='black.Horizontal.TProgressbar')
        bar['value'] = 0
        # bar.grid(column=grid[0], row=grid[1])
        bar.pack()
        # self.tab_control.pack()
        return ElementData(bar)

    def add_button(self, grid: tuple[int, int], text: str, command: tp.Callable) -> None:
        btn = ttk.Button(self.window, text=text, state=["enabled"], command=command)
        # self.tab_control.pack()
        # btn.grid(column=grid[0], row=grid[1])
        # print(self.tab_name)
        btn.pack()

    def add_spinbox(self, grid, label_text, default_spin_value: int, increment: int) -> ElementData:
        self.add_label(label_text, grid)
        spin_var = IntVar(value=default_spin_value)
        spin = Spinbox(self.window, from_=0, to=float('inf'), width=10, increment=increment, textvariable=spin_var)
        # spin.grid(column=grid[0], row=grid[1] + 1)
        spin.pack()
        # self.tab_control.pack()
        return ElementData(spin, spin_var)

    def add_input(self, grid, label_text: str, default_value: str) -> ElementData:
        self.add_label(label_text, grid)
        entry_var = StringVar(value=default_value)
        entry = ttk.Entry(self.window, textvariable=entry_var)
        # entry.grid(column=grid[0], row=grid[1] + 1)
        entry.pack()
        # self.tab_control.pack()
        return ElementData(entry, entry_var)

    def create_tab(self):
        tab = ttk.Frame(self.window)
        return tab

    def pack_tab(self, elements: list[tp.Callable]):
        for element, grid_value in zip(elements, range(0,len(elements * 4), 4)):
                element((0, grid_value))