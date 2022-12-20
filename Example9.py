def factorial (n):
    """The factorial of a number indicates the product of that number for all its antecedents"""
    if n ==1:
        return 1
    else:
        return (n * factorial (n-1))

res = int(input("enter a number:"))
if res >= 1:
    print("The factorial of ", res, "is ", factorial (res))