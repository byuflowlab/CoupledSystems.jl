from pyxdsm.XDSM import XDSM, FUNC, IFUNC, SOLVER

x1 = XDSM()

x1.add_system("D1", FUNC, (r"\text{Discipline 1}", r"y_1 = z_1^2 + z_2 + x - 0.2 y_2"))
x1.add_system("D2", FUNC, (r"\text{Discipline 2}", r"y_2 = \sqrt{y_1} + z_1 + z_2"))
x1.add_system("F", FUNC, (r"\text{Objective}", r"f = x^2 + z_1 + y_1 + e^{-y_2}"))
x1.add_system("G1", FUNC, (r"\text{Constraint 1}", r"g_1 = 3.16 - y_1"))
x1.add_system("G2", FUNC, (r"\text{Constraint 2}", r"g_2 = y_2 - 24.0"))

x1.add_input("D1", r"x, z")
x1.add_input("D2", r"z")
x1.add_input("F", r"x, z_1")

x1.connect("D1", "D2", r"y_1")
x1.connect("D2", "D1", r"y_2")
x1.connect("D1", "F", r"y_1")
x1.connect("D2", "F", r"y_2")
x1.connect("D1", "G1", r"y_1")
x1.connect("D2", "G2", r"y_2")

x1.add_output("F", r"f", side="right")
x1.add_output("G1", r"g_1", side="right")
x1.add_output("G2", r"g_2", side="right")
x1.write("sellar-xdsm1")

x2 = XDSM()

x2.add_system("solver", SOLVER, (r"\text{Nonlinear Solver}"))
x2.add_system("D1", IFUNC, (r"\text{Discipline 1}", r"r_1(y) = z_1^2 + z_2 + x - 0.2 y_2 - y_1"))
x2.add_system("D2", IFUNC, (r"\text{Discipline 2}", r"r_2(y) = \sqrt{y_1} + z_1 + z_2 - y_2"))
x2.add_system("F", FUNC, (r"\text{Objective}", r"f = x^2 + z_1 + y_1 + e^{-y_2}"))
x2.add_system("G1", FUNC, (r"\text{Constraint 1}", r"g_1 = 3.16 - y_1"))
x2.add_system("G2", FUNC, (r"\text{Constraint 2}", r"g_2 = y_2 - 24.0"))

x2.add_input("D1", r"x, z")
x2.add_input("D2", r"z")
x2.add_input("F", r"x, z_1")

x2.connect("solver", "D2", r"y")
x2.connect("solver", "D1", r"y")
x2.connect("D1", "solver", r"r_1(y)")
x2.connect("D2", "solver", r"r_2(y)")
x2.connect("solver", "F", r"y")
x2.connect("solver", "G1", r"y_1")
x2.connect("solver", "G2", r"y_2")

x2.add_output("F", r"f", side="right")
x2.add_output("G1", r"g_1", side="right")
x2.add_output("G2", r"g_2", side="right")
x2.write("sellar-xdsm2")

x3 = XDSM()

x3.add_system("solver", SOLVER, (r"\text{Nonlinear Solver}"))
x3.add_system("D", IFUNC, (r"\text{Disciplines 1 and 2}", r"r(y) = \begin{cases} z_1^2 + z_2 + x - 0.2 y_2 - y_1 \\ \sqrt{y_1} + z_1 + z_2 - y_2 \end{cases}"))
x3.add_system("F", FUNC, (r"\text{Objective}", r"f = x^2 + z_1 + y_1 + e^{-y_2}"))
x3.add_system("G1", FUNC, (r"\text{Constraint 1}", r"g_1 = 3.16 - y_1"))
x3.add_system("G2", FUNC, (r"\text{Constraint 2}", r"g_2 = y_2 - 24.0"))

x3.add_input("D", r"x, z")
x3.add_input("F", r"x, z_1")

x3.connect("solver", "D", r"y")
x3.connect("D", "solver", r"r(y)")
x3.connect("solver", "F", r"y")
x3.connect("solver", "G1", r"y_1")
x3.connect("solver", "G2", r"y_2")

x3.add_output("F", r"f", side="right")
x3.add_output("G1", r"g_1", side="right")
x3.add_output("G2", r"g_2", side="right")
x3.write("sellar-xdsm3")

x4 = XDSM()

x4.add_system("D", FUNC, r"\text{Multidisciplinary Analysis}")
x4.add_system("F", FUNC, (r"\text{Objective}", r"f = x^2 + z_1 + y_1 + e^{-y_2}"))
x4.add_system("G1", FUNC, (r"\text{Constraint 1}", r"g_1 = 3.16 - y_1"))
x4.add_system("G2", FUNC, (r"\text{Constraint 2}", r"g_2 = y_2 - 24.0"))

x4.add_input("D", r"x, z")

x4.connect("D", "F", r"y")
x4.connect("D", "G1", r"y_1")
x4.connect("D", "G2", r"y_2")

x4.add_output("F", r"f", side="right")
x4.add_output("G1", r"g_1", side="right")
x4.add_output("G2", r"g_2", side="right")
x4.write("sellar-xdsm4")

x5 = XDSM()

x5.add_system("S", FUNC, r"\text{Sellar Problem}")

x5.add_input("S", r"x, z")

x5.add_output("S", r"f, g", side="right")
x5.write("sellar-xdsm-final")
