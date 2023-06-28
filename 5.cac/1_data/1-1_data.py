def f(x):
    # Define your function here
    # You can modify this function according to your specific requirements
    return x**2 - 3*x + 2

# Calculate and print f(x) for 0 < x < 4
start = 0.0
end = 4.0
step = 0.1

x = start
while x < end:
    result = f(x)
    print("f({:.2f}) = {:.2f}".format(x, result))
    x += step

