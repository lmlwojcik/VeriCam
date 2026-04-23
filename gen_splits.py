import math
import json
import random
import argparse

def main(
        generate_triplet, true_ratio, val_ratio,
        test_ratio, ratio_test_mixed, ratio_test_unique, min_instances,
        total_pairs, seed, camera_info, index_ignore, out_file
    ):

    random.seed(seed)

    with open(camera_info, "r") as fd:
        dts = json.load(fd)

    # Removing unsuitable cameras
    for k, v in dts.items():
        if len(v) < min_instances:
            index_ignore.append(k)
    for idx in index_ignore:
        del dts[idx]
    
    classes_list = list(dts.keys())
    random.shuffle(classes_list)
    n_classes = len(classes_list)

    # Calculating number of cameras
    n_train_cams = math.ceil(n_classes * (1-test_ratio))
    n_test_cams = n_classes - n_train_cams

    cams_test = classes_list[n_train_cams:]
    cams_train = classes_list[:n_train_cams]

    # Calculating number of instances
    n_train = math.ceil(total_pairs * (1-(val_ratio+test_ratio)))
    n_val = math.ceil(total_pairs*val_ratio)
    n_test = total_pairs - (n_train + n_val)

    n_test_mixed = math.ceil(n_test*ratio_test_mixed)
    n_test_unique = math.ceil(n_test*ratio_test_unique)
    n_test_known = n_test - (n_test_mixed + n_test_unique)

    # Creating protocol
    protocol = {
        "config": {
            "type": "triplet" if generate_triplet else "pairs",
            "test_ratio": test_ratio,
            "ratio_cams_test_unique": ratio_test_unique,
            "total_pairs": total_pairs
        },
        "camera_config": {
            "cameras_train": cams_train,
            "cameras_test": cams_test,
        }
    }

    train_instances = []
    val_instances = []
    test_instances_known = []
    test_instances_mixed = []
    test_instances_unique = []

    # Generating pairs / triplets
    if generate_triplet:
        for _ in range(n_train):
            chosen, neg_cam = random.sample(cams_train, 2)
            anchor, positive = random.sample(dts[chosen], 2)
            negative = random.choice(dts[neg_cam])
            train_instances.append((anchor, positive, negative))

        for _ in range(n_val):
            chosen, neg_cam = random.sample(cams_train, 2)
            anchor, positive = random.sample(dts[chosen], 2)
            negative = random.choice(dts[neg_cam])
            val_instances.append((anchor, positive, negative))

        for _ in range(n_test_known):
            chosen, neg_cam = random.sample(cams_train, 2)
            anchor, positive = random.sample(dts[chosen], 2)
            negative = random.choice(dts[neg_cam])
            test_instances_known.append((anchor, positive, negative))

        for _ in range(n_test_mixed):
            true_known = True if random.random() < 0.5 else False
            if true_known:
                chosen = random.choice(cams_train)
                neg_cam = random.choice(cams_test)
            else:
                chosen = random.choice(cams_test)
                neg_cam = random.choice(cams_train)

            anchor, positive = random.sample(dts[chosen], 2)
            negative = random.choice(dts[neg_cam])
            test_instances_mixed.append((anchor, positive, negative))

        for _ in range(n_test_unique):
            chosen, neg_cam = random.sample(cams_test, 2)
            anchor, positive = random.sample(dts[chosen], 2)
            negative = random.choice(dts[neg_cam])
            test_instances_unique.append((anchor, positive, negative))

    else:
        for _ in range(n_train):
            positive_sample = True if random.random() < true_ratio else False
            chosen, neg_cam = random.sample(cams_train, 2)
            anchor, other = random.sample(dts[chosen], 2)
            if not positive_sample:
                other = random.choice(dts[neg_cam])
            train_instances.append((anchor, other, positive_sample))

        for _ in range(n_val):
            positive_sample = True if random.random() < true_ratio else False
            chosen, neg_cam = random.sample(cams_train, 2)
            anchor, other = random.sample(dts[chosen], 2)
            if not positive_sample:
                other = random.choice(dts[neg_cam])
            val_instances.append((anchor, other, positive_sample))

        for _ in range(n_test_known):
            positive_sample = True if random.random() < true_ratio else False
            chosen, neg_cam = random.sample(cams_train, 2)
            anchor, other = random.sample(dts[chosen], 2)
            if not positive_sample:
                other = random.choice(dts[neg_cam])
            test_instances_known.append((anchor, other, positive_sample))

        for _ in range(n_test_mixed):
            true_known = True if random.random() < 0.5 else False
            if true_known:
                chosen = random.choice(cams_train)
                neg_cam = random.choice(cams_test)
            else:
                chosen = random.choice(cams_test)
                neg_cam = random.choice(cams_train)

            positive_sample = True if random.random() < true_ratio else False
            anchor, other = random.sample(dts[chosen], 2)
            if not positive_sample:
                other = random.choice(dts[neg_cam])
            test_instances_mixed.append((anchor, other, positive_sample))

        for _ in range(n_test_unique):
            positive_sample = True if random.random() < true_ratio else False
            chosen, neg_cam = random.sample(cams_test, 2)
            anchor, other = random.sample(dts[chosen], 2)
            if not positive_sample:
                other = random.choice(dts[neg_cam])
            test_instances_unique.append((anchor, other, positive_sample))

    protocol['data'] = {
        "train_instances": train_instances,
        "val_instances": val_instances,
        "test_instances_known": test_instances_known,
        "test_instances_mixed": test_instances_mixed,
        "test_instances_unique": test_instances_unique
    }

    with open(out_file, "w") as fd:
        json.dump(protocol, fd, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--generate_triplet", action='store_true', default=False)
    parser.add_argument("--true_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.4)
    parser.add_argument("--ratio_test_mixed", type=float, default=0.1)
    parser.add_argument("--ratio_test_unique", type=float, default=0.7)
    parser.add_argument("--min_instances", type=int, default=2)

    parser.add_argument("--total_pairs", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--camera_info", type=str, default='datasets/camera_info.json')
    parser.add_argument("--index_ignore", type=str, nargs='+', default=[])
    parser.add_argument("--out_file", type=str, default='datasets/protocol.json')

    args = vars(parser.parse_args())
    if args['config'] is not None:
        with open(args['config'], "r") as fd:
            cfg = json.load(fd)
    else:
        del args['config']
        cfg = args

    if args['generate_triplet'] and args['true_ratio'] is not None:
        print("Error: triplets generate both true and positive cases.\n" + 
              "<true_ratio> must not be set.")
        exit()
    elif not args['generate_triplet'] and args['true_ratio'] is None:
        print("Error: one of <generate_triplet> or <true_ratio> must be set.")
        exit()

    main(**cfg)
