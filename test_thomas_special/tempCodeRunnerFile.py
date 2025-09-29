        if i == len(x) - 2:
            c[i] = 0
            d[i] = (obs_i[i]+obs_i[i-1])/(2 + c[i-1])
            continue