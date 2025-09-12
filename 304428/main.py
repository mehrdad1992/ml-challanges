from solution import Solution


def main():
    api_key = "trb-2380b1473118a24ba5-d8a1-4b44-b949-432713fdd5e9"
    solution = Solution(api_key=api_key)

    store_name = input()
    product_name = input()
    text = store_name + "\n" + product_name

    print(solution.run(text))


if __name__ == "__main__":
    main()
