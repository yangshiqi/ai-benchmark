
import ai_benchmark
import os, sys


def main():
    benchmark = ai_benchmark.AIBenchmark()
    #results = benchmark.run_training(precision="high")
    #results = benchmark.run_inference(precision="normal")

    #run_inference + run_training on precision="normal"
    results = benchmark.run()



if __name__ == "__main__":
    if not os.path.exists(os.getcwd() + "/ai_benchmark/config.py"):
        sys.exit("ai_benchmark/config.py does not exist. Please make your config.py first.")

    # execute only if run as a script
    main()
