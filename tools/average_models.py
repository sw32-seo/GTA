#!/usr/bin/env python
import argparse
import torch
from tqdm import tqdm


def average_models(avg_opt):
    vocab = None
    opt = None
    avg_model = None
    avg_generator = None

    for step_size in tqdm(range(avg_opt.step_lower, (avg_opt.step_higher + avg_opt.step_size), avg_opt.step_size)):
        model_file = '%s/model_step_%d.pt' % (avg_opt.models_path, step_size)
        m = torch.load(model_file, map_location='cpu')
        model_weights = m['model']
        generator_weights = m['generator']

        if step_size == avg_opt.step_lower:
            vocab, opt = m['vocab'], m['opt']
            avg_model = model_weights
            avg_generator = generator_weights
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(step_size).add_(model_weights[k]).div_(step_size + 1)

            for (k, v) in avg_generator.items():
                avg_generator[k].mul_(step_size).add_(generator_weights[k]).div_(step_size + 1)

    final = {"vocab": vocab, "opt": opt, "optim": None,
             "generator": avg_generator, "model": avg_model}
    return final


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-models_path", help="Directory of models")
    parser.add_argument("-step_lower", type=int, help="Lower bound of step")
    parser.add_argument("-step_higher", type=int, help="Upper bound of step")
    parser.add_argument("-step_size", type=int, help="The step size of model file")
    parser.add_argument("-output", "-o", required=True,
                        help="Output file")
    opt = parser.parse_args()

    final = average_models(opt)
    torch.save(final, opt.output)


if __name__ == "__main__":
    main()
