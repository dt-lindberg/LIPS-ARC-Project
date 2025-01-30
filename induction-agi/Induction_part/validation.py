from enum import Enum
from execute import multi_execute_transformation
import numpy as np


class GridComparisonResult(Enum):
    EQUAL = 0
    SHAPE_MISMATCH = 1
    CONTENT_MISMATCH = 2
    TYPE_MISMATCH = 3
    ERROR = 4
    NON_2D_ARRAY = 5

def compare_grids(output_grid, expected_output_grid):
    if isinstance(output_grid, str):
        return GridComparisonResult.ERROR, 0.0
    
    if not isinstance(output_grid, np.ndarray):
        return GridComparisonResult.TYPE_MISMATCH, 0.0
    
    if len(output_grid.shape) != 2:
        return GridComparisonResult.NON_2D_ARRAY, 0.0
    
    if output_grid.shape != expected_output_grid.shape:
        return GridComparisonResult.SHAPE_MISMATCH, 0.0
    
    if np.array_equal(output_grid, expected_output_grid):
        return GridComparisonResult.EQUAL, 1.0
    
    # If shapes match but content doesn't, calculate the ratio of matching elements
    ratio = np.sum(output_grid == expected_output_grid) / np.prod(expected_output_grid.shape)
    return GridComparisonResult.CONTENT_MISMATCH, ratio


def multi_validate(arc_problem, codes):

    # first execute the first input for each code to filter, leave only the correct ones
    
    results = [list() for _ in range(len(codes))]
    pairs = arc_problem.train_pairs + arc_problem.test_pairs
    for pair_idx in range(len(pairs)):
        input_grid = pairs[pair_idx].x
        try:
            output_grids = multi_execute_transformation(codes, [input_grid]*len(codes), random_seeds=[0]*len(codes),
                                                        timeout=2, function_name="transform", num_workers=64)
        except KeyboardInterrupt:
            exit()

        assert len(output_grids) == len(codes)
        
        for code_idx, output_grid in enumerate(output_grids):
            # compare
            try:
                comparison_result, ratio = compare_grids(output_grid, pairs[pair_idx].y)
            except:
                breakpoint()
            if comparison_result == GridComparisonResult.EQUAL:
                results[code_idx].append((comparison_result == GridComparisonResult.EQUAL, ratio))
            elif comparison_result == GridComparisonResult.SHAPE_MISMATCH:
                results[code_idx].append((comparison_result == GridComparisonResult.EQUAL, ratio))
            elif comparison_result == GridComparisonResult.CONTENT_MISMATCH:
                results[code_idx].append((comparison_result == GridComparisonResult.EQUAL, ratio))
            else:
                results[code_idx].append((None, 0.0))

        assert len(results) == len(codes)

    return results, output_grids

def validate(arc_problem, code, TRANSPOSE=False):
    failure = False

    return_output_grids = []
    train_verdict = False
    for idx, train_pair in enumerate(arc_problem.train_pairs + arc_problem.test_pairs):
        
        if failure: break

        if idx >= len(arc_problem.train_pairs):
            train_verdict = True

        # transpose the input and output grids, because we index them x,y and they are stored as r,c

        if TRANSPOSE:
            input_grid = train_pair.x.T
            expected_output_grid = train_pair.y.T
        else:
            input_grid = train_pair.x
            expected_output_grid = train_pair.y

        try:
            output_grids = multi_execute_transformation([code], [input_grid], random_seeds=[0], timeout=2, 
                                                        function_name="transform", num_workers=32)
            output_grid = output_grids[0]
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            output_grid = "error"
            print(e)


        comparison_result, ratio = compare_grids(output_grid, expected_output_grid)
        
        if isinstance(output_grid, np.ndarray):
            return_output_grids.append(output_grid.tolist())
        else:
            return_output_grids.append(output_grid)

        return_output_grids.append(output_grid.tolist())

        if comparison_result != GridComparisonResult.EQUAL:
            failure = True
            if comparison_result == GridComparisonResult.ERROR:
                print(f"\t\t[-] Error occurred: {output_grid}")
            elif comparison_result == GridComparisonResult.TYPE_MISMATCH:
                print("\t\t[-] output is not a numpy array")
            elif comparison_result == GridComparisonResult.SHAPE_MISMATCH:
                print(f"\t\t[-] output shape does not match expected shape: {output_grid.shape} vs {expected_output_grid.shape}")
            elif comparison_result == GridComparisonResult.CONTENT_MISMATCH:
                print(f"\t\t[-] comparison failed, ratio of correct elements: {ratio}")

    if not failure: print(f"\t[+] passed")

    # if not failure and not train_verdict:
    #     print("something wrong")
    #     exit()

    return (train_verdict, not failure, return_output_grids)