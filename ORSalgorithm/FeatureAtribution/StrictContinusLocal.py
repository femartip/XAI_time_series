def find_continues_x_length(x_y_c: List[Tuple[int, float, int]], target_class: int) -> Dict[float, Tuple[int, int]]:
    """
    Curr not used, but might be later
    Return {x: (min_y,max_y)} s.t. c(x,min_y) = c(x,max_y) = c(x,y') for all y, min_y<=y<=max_y
    :param target_class:
    :param x_y_c:
    :return:
    """
    group_by_x_yc = {}
    for x, y, c in x_y_c:
        if x not in group_by_x_yc:
            group_by_x_yc[x] = []
        group_by_x_yc[x].append((y, c))

    y_min_max_per_x = {}
    for x in group_by_x_yc.keys():
        yc_sorted_by_y = sorted(group_by_x_yc[x], key=lambda yc: yc[0])
        mid = len(yc_sorted_by_y) // 2

        bottom_half = yc_sorted_by_y[:mid]
        idx_bot = mid - 1

        while idx_bot >= 0 and bottom_half[idx_bot][1] == target_class:
            idx_bot -= 1
        if idx_bot == mid - 1:
            min_y_correct_class = bottom_half[idx_bot][0]
        else:
            min_y_correct_class = bottom_half[idx_bot + 1][0]

        top_half = yc_sorted_by_y[mid:]
        idx_top = 0  # mid + idx_top
        while idx_top < len(top_half) and top_half[idx_top][1] == target_class:
            idx_top += 1

        if idx_top == 0:
            max_y_correct_class = top_half[idx_top][0]  # TODO: Make better choice?
        else:
            max_y_correct_class = top_half[idx_top - 1][0]  # Shows last point we had correct

        y_min_max_per_x[x] = (min_y_correct_class, max_y_correct_class)

    return y_min_max_per_x
