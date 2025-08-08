from dataclasses import dataclass, field
import random

@dataclass
class Criterion:
    char : str
    filelist : str
    excl_terms : list[str] = field(default_factory=list)
    req_terms : list[str] = field(default_factory=list)
    or_terms : list[str] = field(default_factory=list)
    max_lines_override : int = None

def process_criteria(
    criteria : list[Criterion],
    out_file : str,
    n_max_lines : int = 200,
    seed : int = 42):

    # criteria = sorted(criteria, key=lambda x: x.char.lower())
    spk_map = dict()
    out_lines = []

    for i,c in enumerate(criteria):
        spk_map[i] = c.char
        
        with open(c.filelist, encoding='utf-8') as f:
            lines = f.readlines()

        random.seed(seed)
        random.shuffle(lines)
        lines_count = 0
        for line in lines:
            if n_max_lines is not None and lines_count >= n_max_lines:
                break
            if c.max_lines_override is not None and lines_count >= c.max_lines_override:
                break

            line = line.strip()
            spl = line.split('|')
            if len(spl) != 3:
                continue
            line_file, char, trancsr = spl
            if char != c.char:
                continue

            if len(c.excl_terms) and any(
                excl_term in line for excl_term in c.excl_terms):
                continue
            if len(c.req_terms) and not all(
                req_term in line for req_term in c.req_terms):
                continue
            if len(c.or_terms) and not any(
                or_term in line for or_term in c.or_terms):
                continue

            out_lines.append(f'{line_file}|{i}')
            lines_count += 1
        
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
    return spk_map

def anchor_recipe(char):
    process_criteria(
        [
        Criterion(
            char=f'{char}_Sing', filelist='ppp_sing_filelist.txt',
            excl_terms=['Studio Recording', '_Very Noisy_']),
        Criterion(
            char=f'{char}_Sing', filelist='ppp_sing_filelist.txt', 
            req_terms=['Studio Recording']),
        Criterion(
            char=char, filelist='ppp_filelist.txt', 
                excl_terms=['_Noisy_', '_Very Noisy_', 'CAUTION']),
        Criterion(
            char='me', filelist='me_filelist.txt', max_lines_override=500),
        Criterion( # Anchor speaker
            char='ex02', filelist='expresso_read_filelist.txt', excl_terms=['longform'], max_lines_override=500),
        Criterion( # Anchor speaker
            char='ex04', filelist='expresso_read_filelist.txt', excl_terms=['longform'], max_lines_override=500),
        ],
        out_file = f'{char.lower().replace(" ", "_")}_anchor.txt'
    )

def no_anchor_recipe(char):
    process_criteria(
        [
        Criterion(
            char=f'{char}_Sing', filelist='ppp_sing_filelist.txt',
            excl_terms=['Studio Recording', '_Very Noisy_']),
        Criterion(
            char=f'{char}_Sing', filelist='ppp_sing_filelist.txt', 
            req_terms=['Studio Recording']),
        Criterion(
            char=char, filelist='ppp_filelist.txt', 
                excl_terms=['_Noisy_', '_Very Noisy_', 'CAUTION']),
        ],
        out_file = f'{char.lower().replace(" ", "_")}_no_anchor.txt'
    )

# process_criteria(
#     [
#         Criterion(
#             char=f'Fluttershy_Sing', filelist='ppp_sing_filelist.txt', 
#             excl_terms=['_Very Noisy_', 'Studio Recording']),
#     ], out_file='fluttershy_sing.txt',
#     n_max_lines=None
# )

# process_criteria(
#     [
#         Criterion(
#             char='ex01', filelist='expresso_read_filelist.txt', excl_terms=['longform'], max_lines_override=500),
#         Criterion(
#             char='ex02', filelist='expresso_read_filelist.txt', excl_terms=['longform'], max_lines_override=500),
#         Criterion(
#             char='ex03', filelist='expresso_read_filelist.txt', excl_terms=['longform'], max_lines_override=500),
#         Criterion(
#             char='ex04', filelist='expresso_read_filelist.txt', excl_terms=['longform'], max_lines_override=500),
#         Criterion(
#             char='me', filelist='me_filelist.txt', max_lines_override=500),
#         *[Criterion(
#             char=x, filelist='ppp_sing_filelist.txt', 
#             req_terms=['Studio Recording'], max_lines_override=500) for x in [
#                 'Fluttershy_Sing',
#                 'Pinkie_Sing',
#                 'Twilight_Sing',
#                 'Rarity_Sing',
#                 'Applejack_Sing',
#                 'Rainbow_Sing',
#                 'Sunset Shimmer_Sing',
#                 'Flash Sentry_Sing',
#             ]],
#         *[Criterion(
#             char=f'p{i}', filelist='vctk_named_filelist.txt', max_lines_override=200)
#             for i in range(225, 376)],
#         *[Criterion(
#             char=x, filelist='ppp_filelist.txt', 
#             req_terms=['Special source'], excl_terms=['REVERB'], max_lines_override=500) for x in [
#                 'Twilight', 'Applejack', 'Rainbow',
#                 'Spike', 'Apple Bloom', 'Fluttershy',
#                 'Pinkie', 'Rarity', 'Starlight', 'Trixie',
#                 'Scootaloo', 'Discord', 'Sunset Shimmer',
#                 'Celestia', 'Sweetie Belle', 'Chrysalis',
#                 'Goldie Delicious', 'Autumn Blaze', 'Mrs. Cake', 'Tirek',
#                 'Big Mac', 'Granny Smith', 'Cozy Glow', "Hoo'far", 'Pear Butter',
#                 'Burnt Oak', 'Sugar Belle'
#             ]]
#     ], out_file='super_diverse2.txt',
#     n_max_lines=None
# )

# process_criteria(
#     [
#         Criterion(
#             char=f'Derpy', filelist='ppp_filelist.txt', 
#             req_terms=['Special source']),
#     ], out_file='derpy_special.txt',
#     n_max_lines=None
# )


process_criteria(
    [
        Criterion(
            char=f'Pinkie', filelist='ppp_filelist.txt', 
            excl_terms=['REVERB', '_Very Noisy_']),
    ], out_file='pinkie_spk.txt',
    n_max_lines=None
)