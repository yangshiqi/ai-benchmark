
import ai_benchmark


def main():
    benchmark = ai_benchmark.AIBenchmark()
    #results = benchmark.run_training(precision="high")
    #results = benchmark.run_inference(precision="normal")

    #run_inference + run_training on precision="normal"
    results = benchmark.run()



if __name__ == "__main__":
    # execute only if run as a script
    main()
