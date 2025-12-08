import os
import time
import psutil
import GPUtil
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


class TrainingMonitor:
    """训练资源监控器"""

    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.logs = []
        self.start_time = time.time()

    def get_system_stats(self):
        """获取系统统计信息"""
        # GPU信息
        gpus = GPUtil.getGPUs()
        gpu_stats = []
        for gpu in gpus:
            gpu_stats.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': f'{gpu.load*100:.1f}%',
                'memory_used': f'{gpu.memoryUsed}/{gpu.memoryTotal}MB',
                'memory_util': f'{gpu.memoryUtil*100:.1f}%',
                'temperature': f'{gpu.temperature}°C'
            })

        # CPU和内存
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()

        stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_time': f"{(time.time() - self.start_time)/60:.1f}min",
            'cpu': f'{cpu_percent}%',
            'memory': f'{memory.percent}%',
            'memory_used': f'{memory.used/1024**3:.1f}/{memory.total/1024**3:.1f}GB',
            'disk_read': f'{disk_io.read_bytes/1024**2:.1f}MB',
            'disk_write': f'{disk_io.write_bytes/1024**2:.1f}MB',
            'gpus': gpu_stats
        }

        return stats

    def print_stats(self, stats):
        """打印统计信息"""
        print(f"\r[{stats['timestamp']}] "
              f"CPU: {stats['cpu']} | "
              f"RAM: {stats['memory']} | "
              f"时间: {stats['elapsed_time']}", end='')

        # GPU信息
        for gpu in stats['gpus']:
            print(f" | GPU{gpu['id']}: {gpu['load']} | "
                  f"显存: {gpu['memory_used']} ({gpu['memory_util']}) | "
                  f"温度: {gpu['temperature']}", end='')

        print()

    def save_logs(self, filename='training_monitor.csv'):
        """保存监控日志"""
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"\n监控日志已保存到: {filename}")

    def plot_resource_usage(self, filename='resource_usage.png'):
        """绘制资源使用图"""
        if not self.logs:
            print("没有数据可以绘制")
            return

        # 提取数据
        times = [log['elapsed_time'] for log in self.logs]
        cpu_usage = [float(log['cpu'].strip('%')) for log in self.logs]
        memory_usage = [float(log['memory'].strip('%')) for log in self.logs]

        # GPU使用率
        gpu_usage = []
        gpu_memory = []
        for log in self.logs:
            if log['gpus']:
                gpu_usage.append(float(log['gpus'][0]['load'].strip('%')))
                gpu_memory.append(float(log['gpus'][0]['memory_util'].strip('%')))
            else:
                gpu_usage.append(0)
                gpu_memory.append(0)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # CPU和内存使用率
        ax1.plot(times, cpu_usage, label='CPU', linewidth=2)
        ax1.plot(times, memory_usage, label='RAM', linewidth=2)
        ax1.set_ylabel('使用率 (%)')
        ax1.set_title('系统资源使用率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # GPU使用率
        ax2.plot(times, gpu_usage, label='GPU计算', linewidth=2, color='green')
        ax2.plot(times, gpu_memory, label='GPU显存', linewidth=2, color='orange')
        ax2.set_xlabel('时间 (分钟)')
        ax2.set_ylabel('使用率 (%)')
        ax2.set_title('GPU资源使用率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"资源使用图已保存到: {filename}")

    def monitor(self, duration=None):
        """开始监控"""
        print("开始监控训练资源使用情况...")
        print("按 Ctrl+C 停止监控")
        print("-" * 80)

        try:
            while True:
                stats = self.get_system_stats()
                self.logs.append(stats)
                self.print_stats(stats)
                time.sleep(self.log_interval)

                if duration and (time.time() - self.start_time) > duration * 60:
                    break

        except KeyboardInterrupt:
            print("\n监控停止")

        # 保存结果
        self.save_logs()
        self.plot_resource_usage()


def main():
    monitor = TrainingMonitor(log_interval=5)  # 每5秒更新一次

    # 监控1小时（可选）
    # monitor.monitor(duration=60)

    # 持续监控直到手动停止
    monitor.monitor()


if __name__ == '__main__':
    main()