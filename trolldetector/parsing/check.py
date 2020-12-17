import argparse


# check if integer is greater than zero
def positive_integer(value):
    arg = int(value)
    if arg <= 0:
        raise argparse.ArgumentTypeError("%s is not an accepted input. must be positive" % arg)

    return arg


# check if input is in (0, 1)
def open_interval(value):
    arg = float(value)
    if arg <= 0 or arg >= 1:
        raise argparse.ArgumentTypeError("%s is not an accepted input. must be in interval (0, 1)" % arg)
    return arg


# check if input is in (0, 1]
def halfopen_interval(value):
    arg = float(value)
    if arg <= 0 or arg > 1:
        raise argparse.ArgumentTypeError("%s is not an accepted input. must be in interval (0, 1]" % arg)
    return arg


# check if input won't set test samples lower than 1
def test(value):
    arg = float(value)
    if arg < 1.6e-06:
        raise argparse.ArgumentTypeError("Value for test set share can not be lower than 1.6e-06")
    return arg


# warn if input will probably lead to high runtimes
def dim_warning(value):
    arg = positive_integer(value)
    if arg >= 30:
        print("Warning: high dimension value will probably lead to high runtime. proceed anyway? [y/n] ", end="")
        check_choice()
    return arg


def k_warning(value):
    arg = positive_integer(value)
    if arg >= 75:
        print("Warning: high k value will probably lead to high runtime. proceed anyway? [y/n] ", end="")
        check_choice()

    return arg


def ngram_warning(value):
    arg = positive_integer(value)
    if arg >= 3:
        print("Warning: ngram-range higher than (1,2) will probably lead to segmentation fault. proceed anyway? [y/n] ",
              end="")
        check_choice()
    return arg


def tol_warning(value):
    arg = open_interval(value)
    if arg <= 1e-05:
        print("Warning: MLP tolerance lower than 1e-05 will probably lead to high runtime. proceed anyway? [y/n] ",
              end="")
        check_choice()
    return arg


# ask user to confirm selected arguments
def check_choice():
    proceed = False
    indifferent = 0
    while not proceed:
        answer = str(input())
        if answer.lower() in ["y", "yes", "j", "ja"]:
            proceed = True
        elif answer.lower() in ["n", "no", "nein"] or indifferent >= 2:
            exit(0)
        else:
            print("\n[y/n] ", end="")
            indifferent += 1
