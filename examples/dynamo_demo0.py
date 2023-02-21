from paddlefx._eval_frame import set_eval_frame


def callback(frame):
    print('enter callback')


def add(a, b):
    return a + b


set_eval_frame(callback)

add(1, 3)

set_eval_frame(None)

print('fin')
