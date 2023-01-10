import filters
import torch_filters

# from filters import test_imp

# from Ensrf import ensrf


def main():
    filters.senkf()
    torch_filters.ensrf_torch()

    print("import successful")


if __name__ == "__main__":
    main()
