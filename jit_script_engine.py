import torch

def train_loop(args, model, optim_func, data_func) :
    model.to(device=args.device)
    jit_model = torch.jit.script(model)
   
    optimizer = optim_func(model.parameters())

    batches = data_func(args.steps, args.data_dtype, args.device)

    with torch.jit.fuser('fuser2') :
        for step,batch in enumerate(batches) :
            loss = jit_model(batch)
            loss.backward()
 
            if step % args.grad_accum_steps == 0 :
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
