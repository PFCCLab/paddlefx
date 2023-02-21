from paddlefx._eval_frame import set_eval_frame


def callback(frame):
    print('enter callback')


def add(a, b):
    print('call add')
    return a + b


print('start set_eval_frame(callback)')
set_eval_frame(callback)
print('fin set_eval_frame(callback)')

add(1, 3)

print('\nstart set_eval_frame(None)')
set_eval_frame(None)
print('fin set_eval_frame(None)')

add(1, 4)

print('\nstart set_eval_frame(callback)')
set_eval_frame(callback)
print('fin set_eval_frame(callback)')

add(1, 3)

print('\nfin')
