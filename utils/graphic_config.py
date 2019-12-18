from graphics import GraphWin, Entry, Point, Text, Rectangle
import yaml
from threading import Thread
from utils import methaboard

config = None
mb = None


def load_logs(config_file):
    global config
    with open(config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)


def write_logs(config_file):
    global config
    with open(config_file, "w+") as log_file:
        yaml.safe_dump(config, log_file, default_flow_style=False)


def button(ll_point, ur_point, text, color="white"):
    but = Rectangle(Point(ll_point[0], ll_point[1]), Point(ur_point[0], ur_point[1]))  # points are ordered ll, ur
    but.setFill(color)

    text = Text(Point((ll_point[0] + ur_point[0]) // 2, (ll_point[1] + ur_point[1]) // 2), text)
    but.draw(win)
    text.draw(win)

    return but, text


def push_button(name, but):
    global origin_logs_path
    if name == "reload":
        config["active"] = False
        but[0].setFill("red")
        load_logs(origin_logs_path)
    else:
        if config[name]:
            config[name] = False
            but[0].setFill("red")
        else:
            config[name] = True
            but[0].setFill("green")


def modify_button(but):
    global selection_types, mutation_types, iteration_types
    name = but[1].getText()

    if name in selection_types:
        index = selection_types.index(name)
        new_index = (index + 1) % len(selection_types)
        but[1].setText(selection_types[new_index])
        config["selection_type"] = selection_types[new_index]

    if name in mutation_types:
        index = mutation_types.index(name)
        new_index = (index + 1) % len(mutation_types)
        but[1].setText(mutation_types[new_index])
        config["mutation_type"] = mutation_types[new_index]

    if name in iteration_types:
        index = iteration_types.index(name)
        new_index = (index + 1) % len(iteration_types)
        but[1].setText(iteration_types[new_index])
        config["iteration_type"] = iteration_types[new_index]


def init_color(name):
    if config[name]:
        return "green"
    else:
        return "red"


def make_text(x, y, size, text):
    text = Text(Point(x, y), text)
    text.setSize(size)
    text.draw(win)
    return text


def make_text_with_input(x, y, col, size, text, init_value="", width=10):
    begin = 0
    if col == 1:
        begin = 220
    elif col == 2:
        begin = 580
    elif col == 3:
        begin = 920
    elif col == 4:
        begin = 1180
    text = make_text(x, y, size, text)
    entry = Entry(Point(begin, y), width)
    entry.setFill("white")
    entry.setText(init_value if init_value != "None" else "")
    entry.draw(win)
    return entry, text


def inside(point, button):
    """ Is point inside rectangle? """
    rectangle = button[0]

    ll = rectangle.getP1()  # assume p1 is ll (lower left)
    ur = rectangle.getP2()  # assume p2 is ur (upper right)

    return ll.getX() < point.getX() < ur.getX() and ll.getY() > point.getY() > ur.getY()


def refresh_config():
    global float_items, tmp_logs_path
    for entry in entries:
        name = entry[1].getText()
        if name == "num of new indiv.:":
            name = "num_of_new_individual"
        else:
            name = name.replace(' ', '_')
            name = name[:-1]

        if name in float_items:
            config[name] = None if entry[0].getText() == '' else float(entry[0].getText())
        else:
            config[name] = None if entry[0].getText() == '' else int(entry[0].getText())

    write_logs(tmp_logs_path)


def run_methaboard():
    global mb

    if mb is None:
        mb = methaboard.MethaBoard()
        path = "./logs/" + methaboard_entry[0].getText() + "/log"
        thread = Thread(target=mb.run, args=(path, ))
        thread.setDaemon(False)
        thread.start()

        methaboard_button[0].undraw()
        methaboard_button[1].undraw()
        methaboard_entry[0].undraw()
        methaboard_entry[1].undraw()


origin_logs_path = "./config.yml"
tmp_logs_path = "./config_tmp.yml"

load_logs(tmp_logs_path)

selection_types = ["random", "tournament", "better half"]
mutation_types = ["basic", "bacterial"]
iteration_types = ["pso", "sso"]
float_items = ["CR", "F", "sigma_init", "sigma_fin", "mutation_probability", "step_size", "norm", "inertia", "phi_p", "phi_g", "c_w", "c_p", "c_g"]

entries = []

win = GraphWin("MethaConfig", 1300, 720)

active_button = button((50, 700), (300, 550), "ACTIVE", init_color("active"))
stop_button = button((400, 700), (650, 600), "STOP", init_color("stop"))
reload_origin_button = button((400, 570), (650, 470), "RELOAD CONFIG", "yellow")
exit_button = button((1080, 700), (1280, 630), "EXIT", "blue")

make_text(110, 20, 16, "Population and other")

population_size_entry = make_text_with_input(75, 50, 1, 14, "population size:", str(config["population_size"]))
chromosome_size_entry = make_text_with_input(88, 80, 1, 14, "chromosome size:", str(config["chromosome_size"]))
pool_button = button((12, 128), (132, 100), "pool", init_color("pool"))
pool_size_entry = make_text_with_input(51, 145, 1, 14, "pool size:", str(config["pool_size"]))
entries += [population_size_entry, chromosome_size_entry, pool_size_entry]

make_text(85, 193, 16, "Stop conditions")

max_iteration_entry = make_text_with_input(68, 223, 1, 14, "max iteration:", str(config["max_iteration"]))
max_fitness_eval_entry = make_text_with_input(81, 253, 1, 14, "max fitness eval:", str(config["max_fitness_eval"]))
min_fitness_entry = make_text_with_input(61, 283, 1, 14, "min fitness:", str(config["min_fitness"]))
patience_entry = make_text_with_input(52, 313, 1, 14, "patience:", str(config["patience"]))
entries += [max_iteration_entry, max_fitness_eval_entry, min_fitness_entry, patience_entry]

make_text(485, 20, 16, "Modify all individuals methods")

mutation_button = button((343, 67), (463, 39), "mutation", init_color("mutation"))
mutation_type_button = button((343, 102), (463, 74), str(config["mutation_type"]), "white")
elitism_button = button((343, 137), (405, 109), "elitism", init_color("elitism"))
mrs_button = button((343, 172), (405, 144), "mrs", init_color("mutation_random_sequence"))
mutation_prob_entry = make_text_with_input(425, 191, 2, 14, "mutation probability:", str(config["mutation_probability"]))
num_of_clones_entry = make_text_with_input(403, 221, 2, 14, "num of clones:", str(config["num_of_clones"]))
entries += [mutation_prob_entry, num_of_clones_entry]

memetic_button = button((343, 275), (463, 247), "memetic", init_color("memetic"))
lrs_button = button((343, 313), (463, 285), "lrs", init_color("lamarck_random_sequence"))
step_size_entry = make_text_with_input(382, 332, 2, 14, "step size:", str(config["step_size"]))
number_of_steps_entry = make_text_with_input(411, 362, 2, 14, "number of steps:", str(config["number_of_steps"]))
entries += [step_size_entry, number_of_steps_entry]


third_col_delta_x = 680
third_col_delta_y = -330
make_text(145 + third_col_delta_x, 350 + third_col_delta_y, 16, "Add new individual methods")

crossover_button = button((12 + third_col_delta_x, 397 + third_col_delta_y), (132 + third_col_delta_x, 369 + third_col_delta_y), "crossover", init_color("crossover"))
selection_type_button = button((12 + third_col_delta_x, 434 + third_col_delta_y), (132 + third_col_delta_x, 404 + third_col_delta_y), str(config["selection_type"]), "white")
num_of_crossover_entry = make_text_with_input(85 + third_col_delta_x, 452 + third_col_delta_y, 3, 14, "num of crossover:", str(config["num_of_crossover"]))

diff_evol_button = button((12 + third_col_delta_x, 498 + third_col_delta_y), (132 + third_col_delta_x, 470 + third_col_delta_y), "diff. evolution", init_color("differential_evolution"))
CR_entry = make_text_with_input(28 + third_col_delta_x, 516 + third_col_delta_y, 3, 14, "CR:", str(config["CR"]))
F_entry = make_text_with_input(22 + third_col_delta_x, 546 + third_col_delta_y, 3, 14, "F:", str(config["F"]))
entries += [num_of_crossover_entry, CR_entry, F_entry]

invasive_weed_button = button((12 + third_col_delta_x, 600 + third_col_delta_y), (132 + third_col_delta_x, 572 + third_col_delta_y), "invasive weed", init_color("invasive_weed"))
iter_max_entry = make_text_with_input(50 + third_col_delta_x, 618 + third_col_delta_y, 3, 14, "iter max:", str(config["iter_max"]))
e_entry = make_text_with_input(23 + third_col_delta_x, 648 + third_col_delta_y, 3, 14, "e:", str(config["e"]))
sigma_init_entry = make_text_with_input(56 + third_col_delta_x, 678 + third_col_delta_y, 3, 14, "sigma init:", str(config["sigma_init"]))
sigma_fin_entry = make_text_with_input(55 + third_col_delta_x, 708 + third_col_delta_y, 3, 14, "sigma fin:", str(config["sigma_fin"]))
N_min_entry = make_text_with_input(42 + third_col_delta_x, 738 + third_col_delta_y, 3, 14, "N min:", str(config["N_min"]))
N_max_entry = make_text_with_input(45 + third_col_delta_x, 768 + third_col_delta_y, 3, 14, "N max:", str(config["N_max"]))
entries += [iter_max_entry, e_entry, sigma_init_entry, sigma_fin_entry, N_min_entry, N_max_entry]

add_new_button = button((12 + third_col_delta_x, 822 + third_col_delta_y), (132 + third_col_delta_x, 794 + third_col_delta_y), "add pure new", init_color("add_pure_new"))
num_of_new_entry = make_text_with_input(90 + third_col_delta_x, 840 + third_col_delta_y, 3, 14, "num of new indiv.:", str(config["num_of_new_individual"]))
entries += [num_of_new_entry]


fourth_col_delta_x = 980
fourth_col_delta_y = 20
make_text(145 + fourth_col_delta_x, 0 + fourth_col_delta_y, 16, "Swarm optimization methods")

swarm_iter_button = button((12 + fourth_col_delta_x, 47 + fourth_col_delta_y), (132 + fourth_col_delta_x, 19 + fourth_col_delta_y), "swarm iter", init_color("swarm_iteration"))
iteration_type_button = button((12 + fourth_col_delta_x, 82 + fourth_col_delta_y), (132 + fourth_col_delta_x, 54 + fourth_col_delta_y), str(config["iteration_type"]), "white")

norm_entry = make_text_with_input(37 + fourth_col_delta_x, 100 + fourth_col_delta_y, 4, 14, "norm:", str(config["norm"]))
inertia_entry = make_text_with_input(41 + fourth_col_delta_x, 130 + fourth_col_delta_y, 4, 14, "inertia:", str(config["inertia"]))
phi_p_entry = make_text_with_input(39 + fourth_col_delta_x, 160 + fourth_col_delta_y, 4, 14, "phi_p:", str(config["phi_p"]))
phi_g_entry = make_text_with_input(39 + fourth_col_delta_x, 190 + fourth_col_delta_y, 4, 14, "phi_g:", str(config["phi_g"]))
c_w_entry = make_text_with_input(33 + fourth_col_delta_x, 220 + fourth_col_delta_y, 4, 14, "c_w:", str(config["c_w"]))
c_p_entry = make_text_with_input(33 + fourth_col_delta_x, 250 + fourth_col_delta_y, 4, 14, "c_p:", str(config["c_p"]))
c_g_entry = make_text_with_input(33 + fourth_col_delta_x, 280 + fourth_col_delta_y, 4, 14, "c_g:", str(config["c_g"]))
entries += [norm_entry, inertia_entry, phi_p_entry, phi_g_entry, c_w_entry, c_p_entry, c_g_entry]

# methaboard
methaboard_button = button((12 + third_col_delta_x, 660), (132 + third_col_delta_x, 632), "run methaboard", "yellow")
methaboard_entry = make_text_with_input(728, 680, 3, 14, "log dir:", "", width=30)

while True:
    if config["active"]:
        refresh_config()

    clickPoint = win.getMouse()

    if inside(clickPoint, stop_button):
        push_button("stop", stop_button)
        continue
    if inside(clickPoint, exit_button):
        del mb
        win.close()
        exit()
    if inside(clickPoint, active_button):
        push_button("active", active_button)
        continue
    if inside(clickPoint, pool_button):
        push_button("pool", pool_button)
        continue
    if inside(clickPoint, crossover_button):
        push_button("crossover", crossover_button)
        continue
    if inside(clickPoint, diff_evol_button):
        push_button("differential_evolution", diff_evol_button)
        continue
    if inside(clickPoint, invasive_weed_button):
        push_button("invasive_weed", invasive_weed_button)
        continue
    if inside(clickPoint, add_new_button):
        push_button("add_pure_new", add_new_button)
        continue
    if inside(clickPoint, mutation_button):
        push_button("mutation", mutation_button)
        continue
    if inside(clickPoint, elitism_button):
        push_button("elitism", elitism_button)
        continue
    if inside(clickPoint, mrs_button):
        push_button("mutation_random_sequence", mrs_button)
        continue
    if inside(clickPoint, memetic_button):
        push_button("memetic", memetic_button)
        continue
    if inside(clickPoint, lrs_button):
        push_button("lamarck_random_sequence", lrs_button)
        continue
    if inside(clickPoint, swarm_iter_button):
        push_button("swarm_iteration", swarm_iter_button)
        continue
    if inside(clickPoint, reload_origin_button):
        push_button("reload", active_button)
        continue

    if inside(clickPoint, selection_type_button):
        modify_button(selection_type_button)
        continue
    if inside(clickPoint, mutation_type_button):
        modify_button(mutation_type_button)
        continue
    if inside(clickPoint, iteration_type_button):
        modify_button(iteration_type_button)
        continue

    if inside(clickPoint, methaboard_button):
        run_methaboard()
        continue

