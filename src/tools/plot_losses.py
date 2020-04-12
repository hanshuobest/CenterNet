import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(log_file):

    loss_data = open(log_file)
    all_lines = loss_data.readlines()
    print(all_lines[4].split(' '))
    # losses
    total_loss = []          # 4
    hm_loss = []             # 7
    wh_loss = []             # 10
    off_loss = []            # 13
    val_loss = []            # 19
    spend_time = []          # 16
    num_lines = len(all_lines)
    for line in range(num_lines):
        total_loss1 = all_lines[line].split(' ')[4]
        hm_loss1 = all_lines[line].split(' ')[7]
        wh_loss1 = all_lines[line].split(' ')[10]
        off_loss1 = all_lines[line].split(' ')[13]
        spend_time1 = all_lines[line].split(' ')[16]

        total_loss.append(float(total_loss1))
        hm_loss.append(float(hm_loss1))
        wh_loss.append(float(wh_loss1))
        off_loss.append(float(off_loss1))
        spend_time.append(float(spend_time1))

    index_val = np.linspace(0, 140, 29) - 1
    index_val = np.delete(index_val, 0, 0)

    for id in index_val:

        val_loss1 = all_lines[int(id)].split(' ')[19]
        val_loss.append(float(val_loss1))
    return val_loss, total_loss


if __name__ == '__main__':
    # 标准图形绘制

    vloss_res18, loss_res18 = plot_loss_curve(
        '/home/han/git_project/detection/CenterNet/exp/ctdet/coco_dla/logs_2020-04-12-15-02/log.txt')              # 读取训练时生成的日志文件

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(range(len(loss_res18)), loss_res18, 'c',
            label='res_18_train_loss', linewidth=1)         # 这个label是图线自己的标签；
    
    ax.set_xlabel('epochs')                                    # 设置坐标轴标签；
    ax.set_ylabel('loss_value')
    # ax.text(8750, 20, "海拔", color='red')                     # 加入文本
    ax.set_title('loss_of_CenterNet')
    # 将图例摆放在不遮挡图线的位置即可
    ax.legend(loc='best')
    ax.grid()                                                  # 添加网格
    # 保存文件到指定文件夹
    plt.savefig('loss_of_CenterNet.png')
    # plt.show()

    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(range(len(vloss_res18)), vloss_res18, 'c',
             label='res_18_val_loss', linewidth=2)         # 这个label是图线自己的标签；
    
    ax1.set_xlabel('epochs')                                    # 设置坐标轴标签；
    ax1.set_ylabel('loss_value')
    # ax.text(8750, 20, "海拔", color='red')                     # 加入文本
    ax1.set_title('val_loss_of_CenterNet')
    # 将图例摆放在不遮挡图线的位置即可
    ax1.legend(loc='best')
    ax1.grid()                                                  # 添加网格
    # 保存文件到指定文件夹
    plt.savefig('val_loss_of_CenterNet.png')
    # plt.show()
