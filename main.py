import datetime

def main():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"Current date is: {current_date}")


if __name__ == "__main__":
    main()
