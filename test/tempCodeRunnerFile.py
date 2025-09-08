 # for _ in range(max_iter):
        #     index_point = (int(current_point[0]), int(current_point[1]))
        #     dist_obs = self.field.dist_to_closest(index_point)
        #     dir_obs = self.field.dir_to_closest(index_point)
        #     gx, gy = APF.F_APF(current_point, goal_xy, dist_obs, dir_obs, 2000, 1, self.radius, self.radius * 200)

        #     current_point = (current_point[0]+gx*0.005, current_point[1] + gy*0.005)
        #     path.add_point((current_point[0], current_point[1]))
        #     if dist_between(current_point, goal_xy) < self.radius:
        #         print("her")
        #         return path