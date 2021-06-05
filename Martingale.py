import random
sc = 0
minimum = 0
bet = 2.5
#model.fit()
def o48():
    x = random.randint(0, 36)
    if x > 0 and x < 19:
        return 1
    else:
        return 0
def play():
    global sc, l, minimum, bet
    for i in range(1, 1*1+1):
        x = o48()#random.randint(0,1)
        if x == 0:
            sc += (-bet)
            bet = bet*2
        else:
            sc+=bet
            bet = 2.5
        #print(sc, bet)

        if sc < minimum:
            minimum = sc
        # print('total:', sc)

for _ in range(1, 100+1):
    sc = 0
    minimum = 0
    bet = 2.5
    play()
    attempt = 0
    while sc < 10000:
        attempt+=1
        play()
    print('Попыток:', attempt, 'Дней:', round(attempt/1440, 2), minimum)