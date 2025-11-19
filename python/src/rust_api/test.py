import onebitml

try:
    op = int(input("Enter a number: "))

    result = onebitml.square_odd_reduce_even(op)

    print(f"\nReturn Value: {result}")

except ValueError:
    print("Invalid input.\n")