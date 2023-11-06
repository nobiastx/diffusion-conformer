#!/usr/bin/env python3
import data, database, models, diffusion
import torch
import argparse, os, warnings

# we appreciate the warning, but this is to reduce output clutter
warnings.filterwarnings('ignore', message='scatter_reduce\(\) is in beta and the API may change at any time')


class logger:
    """Logger object for this application
    
    Parameters
    ----------
    path : str
        Log file path
    verbose : bool, default True
        If true, log file output is verbose
    progress : bool, default False
        If true, checkpoint files are written after each epoch
    """
    def __init__(self, path, verbose=True, progress=None):
        self.verbose = verbose
        self.progress = progress
        self.log = open(path,"w")
        self.log.write("step,loss,validation,rate,memory,maxmemory\n")

    def report(self,epoch,loss,validation,optimizer):
        """Report on training statistics for an epoch
        
        Parameters
        ----------
        epoch : int
            The epoch being reported, counting from zero
        loss : float
            The training loss
        validation : float
            The validation loss
        optimizer : torch.optim.Optimizer
            The optimizer object
        """
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        m1 = torch.cuda.memory_allocated()
        m2 = torch.cuda.max_memory_allocated()
        if self.verbose:
            print(f"{epoch:6d} {loss:10.4f} {validation:10.4f} {lr:9.7f} {m1} {m2}")
        self.log.write(f"{epoch},{loss:.6f},{validation:.6f},{lr:.7g},{m1},{m2}\n")
        self.log.flush()

    def save_progress(self,epoch,model):
        """Save progress
        
        Parameters
        ----------
        epoch : int
            The epoch being reported, counting from zero
        model : nn.Module
            The model, in its currently trained state
        """
        if not self.progress is None:
            torch.save(model,os.path.join(self.progress,f"prog.pt"))


def arguments():
    parser = argparse.ArgumentParser("Train diffusion model")
    parser.add_argument("input", type=str, help="Training dataset")
    parser.add_argument("-v","--validation", type=str, default="validate", help="Validation dataset")
    parser.add_argument("-o","--output", type=str, default="output", help="Output directory")
    parser.add_argument("-n","--num_epoch", type=int, default=100, help="Number of training epochs")
    parser.add_argument("-w","--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("-s","--steps", type=int, default=100, help="Number of noising steps")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dim_embedding", type=int, default=50, help="Atom embedding size")
    parser.add_argument("--num_layer", type=int, default=4, help="Number layers")
    parser.add_argument("--dim_feed", type=int, default=None, help="Transformer feed size, if different than dim_embedding")
    parser.add_argument("--num_aggregate_hidden", type=int, default=2, help="Number of hidden layers in final aggregate")
    parser.add_argument("--num_head", type=int, default=5, help="Number heads in transformer")
    parser.add_argument("-t","--train", type=str, default="train", help="Training set name")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("-d","--device", type=str, default="cuda", help="Processing device")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization")
    parser.add_argument("--seed", type=int, default=3221, help="Random seed")
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = arguments()

    if not os.path.exists(args.input):
        raise Exception(f"No file found at {args.input}")

    #
    # Setup trainer object
    #
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    trainer = diffusion.trainer(generator, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    #
    # Prepare training and validation training sets
    #
    dataset = database.Storage(args.input, readOnly=True)

    def make_loader(name):
        conformers = data.Conformers(dataset, name)
        return data.MoleculeDataLoader(
            conformers,
            device='cpu',
            batch_size=args.batch_size, 
            num_workers=args.workers,
            pin_memory=True,
            sampler=data.MoleculeEvenSampler(conformers,seed=args.seed)
        )

    train_loader = make_loader(args.train)
    validation_loader = make_loader(args.validation)

    #
    # Build model
    #
    dim_feed = args.dim_embedding if args.dim_feed is None else args.dim_feed

    model = models.DiffusionTransformer(
        dataset.number_tokens(),
        dim_embedding=args.dim_embedding, 
        num_layer=args.num_layer,
        dim_feed=dim_feed,
        num_head=args.num_head,
        num_aggregate_hidden=args.num_aggregate_hidden,
        dropout=0.0
    ).to(args.device)

    #
    # Train
    #
    os.makedirs(args.output,exist_ok=True)

    log = logger(os.path.join(args.output,"train.log"), progress=args.output)

    trainer.train(model, train_loader, validation_loader, log, num_steps=args.steps, num_epoch=args.num_epoch)

    torch.save(model, os.path.join(args.output,"model.pt"))


