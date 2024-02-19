class Solution():
    def __init__(self):
        self.cost = [2, 4, 5, 9, 12]
        self.revenue = [None, 7, 6, 2, 1]
        self.price = 12

        self.memo = dict()

    def recurse(self, machine_age, time_left):
        if time_left == 0:
            return 0

        key = (machine_age, time_left)
        if key in self.memo:
            print("pruning")
            return self.memo[key]

        if machine_age == 0:
            best = self.cost[machine_age] + self.recurse(machine_age + 1, time_left - 1)
        else:
            best = min(
                self.cost[machine_age] + self.recurse(machine_age + 1, time_left - 1),
                self.price - self.revenue[machine_age] + self.recurse(0, time_left - 1)
            )
        self.memo[key] = best
        if time_left == 5:
            print(self.memo)
        return best

    def optimize(self):
        return self.recurse(0, 5)

print(f"\nThe best cost solution is: {Solution().optimize()} KEuro")