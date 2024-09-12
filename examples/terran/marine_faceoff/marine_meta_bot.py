from examples.terran.marine_faceoff.marine_helper_funcs import *

from examples.terran.marine_faceoff.marine_match_bot import MarineBot

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))


class MarineMetaBot:

    def __init__(self, history_file_name='history_dict', meta_meta_run_number=1):
        # with open("history.csv", mode="w", encoding="utf-8", newline="") as afp:
        #     # TODO: read existing knowledge as if he bot "rose from failure"
        #     pass
        self.history = pd.DataFrame()
        self.repeat_num = 20
        self.history_dict = {}
        self.meta_meta_run_number = meta_meta_run_number
        self.history_file_name = history_file_name

    def run_meta_bot_multiple_games(self):
        value = []
        matches_list = []

        for i, max_allocation in enumerate([1, 2, 3, 4, 8]):
            print(f'this is {i} round')
            for j in range(0, self.repeat_num):

                print(f'this is {j} repeat')
                self.history.loc[i*self.repeat_num+j, 'max_allocation'] = max_allocation
                self.history_dict[i*self.repeat_num+j] = {}
                self.history_dict[i * self.repeat_num + j]['max_allocation'] = max_allocation
                self.history_dict[i * self.repeat_num + j]['arm_run'] = j

                matches_list.append(
                    GameMatch(
                        maps.get("Blistering Sands_marines2"),
                        # maps.get("marines_4x4"), #maps.get("Blistering Sands_marines"),
                        [
                            Bot(
                                Race.Terran,
                                MarineBot(max_allocation=max_allocation, result=value,
                                          history_dict=self.history_dict, run_index=i * self.repeat_num + j,
                                          meta_meta_run_number=self.meta_meta_run_number)
                            ),
                            Computer(Race.Terran, Difficulty.Easy)
                        ],
                        realtime=True,
                        # save_replay_as="ZvT.SC2Replay",
                        disable_fog=True
                    )
                )



            #self.history.loc[i, 'results']
        run_multiple_games(matches_list)

        for i, max_allocation in enumerate([1, 2, 3, 4, 8]):
            print(f'this is {i} round')
            for j in range(0, self.repeat_num):
                print(f'this is {j} repeat')
                print('value is: ' + str(value))
                self.history.loc[i * self.repeat_num + j, 'value'] = value[i * self.repeat_num + j]

        print(self.history)
        self.history.to_csv(f'./data/arms_results{self.meta_meta_run_number}.csv')

        # for i, max_allocation in enumerate([1, 2, 3, 4, 8]):
        #     print(f'this is {i} round')
        #     for j in range(0, self.repeat_num):
        #
        #         print(f'this is {j} repeat')
        #         self.history.loc[i*self.repeat_num+j, 'max_allocation'] = max_allocation
        #         self.history_dict[i*self.repeat_num+j] = {}
        #         self.history_dict[i * self.repeat_num + j]['max_allocation'] = max_allocation
        #         self.history_dict[i * self.repeat_num + j]['arm_run'] = j
        #         time.sleep(10)
        #         run_game(
        #             maps.get("Blistering Sands_marines2"),
        #             # maps.get("marines_4x4"), #maps.get("Blistering Sands_marines"),
        #             [
        #                 Bot(
        #                     Race.Terran,
        #                     MarineBot(max_allocation=max_allocation, result=value,
        #                               history_dict=self.history_dict, run_index=i*self.repeat_num+j,
        #                               meta_meta_run_number=self.meta_meta_run_number)
        #                 ),
        #                 Computer(Race.Terran, Difficulty.Easy)
        #             ],
        #             realtime=True,
        #             # save_replay_as="ZvT.SC2Replay",
        #             disable_fog=True
        #         )
        #         print('value is: '+str(value))
        #         self.history.loc[i*self.repeat_num+j, 'value'] = value[i*self.repeat_num+j]


            #self.history.loc[i, 'results']
        print(self.history)
        self.history.to_csv(f'./data/arms_results{self.meta_meta_run_number}.csv')

        with open(f'./data/{self.history_file_name}{self.meta_meta_run_number}.json', 'w') as fd:
            json.dump(self.history_dict, fd, indent=4)

    def run_meta_bot(self):
        value = []
        for i, max_allocation in enumerate([1, 2, 3, 4, 8]):
            print(f'this is {i} round')
            for j in range(0, self.repeat_num):
                print(f'this is {j} repeat')
                self.history.loc[i * self.repeat_num + j, 'max_allocation'] = max_allocation
                self.history_dict[i * self.repeat_num + j] = {}
                self.history_dict[i * self.repeat_num + j]['max_allocation'] = max_allocation
                self.history_dict[i * self.repeat_num + j]['arm_run'] = j
                time.sleep(10)
                run_game(
                    maps.get("Blistering Sands_marines2"),
                    # maps.get("marines_4x4"), #maps.get("Blistering Sands_marines"),
                    [
                        Bot(
                            Race.Terran,
                            MarineBot(max_allocation=max_allocation, result=value,
                                      history_dict=self.history_dict, run_index=i * self.repeat_num + j,
                                      meta_meta_run_number=self.meta_meta_run_number)
                        ),
                        Computer(Race.Terran, Difficulty.Easy)
                    ],
                    realtime=True,
                    # save_replay_as="ZvT.SC2Replay",
                    disable_fog=True
                )
                print('value is: ' + str(value))
                self.history.loc[i * self.repeat_num + j, 'value'] = value[i * self.repeat_num + j]

            # self.history.loc[i, 'results']
        print(self.history)
        self.history.to_csv(f'./data/arms_results{self.meta_meta_run_number}.csv')

        with open(f'./data/{self.history_file_name}{self.meta_meta_run_number}.json', 'w') as fd:
            json.dump(self.history_dict, fd, indent=4)


def main_meta():
    # run_multiple_games(
    #     matches= [
    #         GameMatch(
    #             maps.get("Blistering Sands_marines2"),
    #                 # maps.get("marines_4x4"), #maps.get("Blistering Sands_marines"),
    #                 [
    #                     Bot(
    #                         Race.Terran,
    #                         MarineBot(max_allocation=2, result=None,
    #                                   history_dict=None, run_index=1,
    #                                   meta_meta_run_number=1)
    #                     ),
    #                     Computer(Race.Terran, Difficulty.Easy)
    #                 ],
    #                 realtime=True,
    #                 # save_replay_as="ZvT.SC2Replay",
    #                 disable_fog=True
    #
    #         ),
    #         GameMatch(
    #             maps.get("Blistering Sands_marines2"),
    #             # maps.get("marines_4x4"), #maps.get("Blistering Sands_marines"),
    #             [
    #                 Bot(
    #                     Race.Terran,
    #                     MarineBot(max_allocation=2, result=None,
    #                               history_dict=None, run_index=1,
    #                               meta_meta_run_number=1)
    #                 ),
    #                 Computer(Race.Terran, Difficulty.Easy)
    #             ],
    #             realtime=True,
    #             # save_replay_as="ZvT.SC2Replay",
    #             disable_fog=True
    #
    #         )
    #     ]
    # )

    meta_bot = MarineMetaBot(meta_meta_run_number=2)
    meta_bot.run_meta_bot_multiple_games()


if __name__ == "__main__":
    #main()
    import time

    start = time.time()
    print("hello")
    main_meta()
    end = time.time()
    print(end - start)
