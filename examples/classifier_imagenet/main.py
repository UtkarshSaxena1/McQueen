import models.imagenet as imagenet_extra_models
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from examples import *
from torchviz import make_dot
best_acc1 = 0


def main():
    parser = get_base_parser()
    args = parser.parse_args()
    hp = get_hyperparam(args)
    if hp.gpu_id == eppb.GPU.ANY:
        args.gpu = get_freer_gpu()
    elif hp.gpu_id == eppb.GPU.NONE:
        args.gpu = None  # TODO: test
     
    
    
    
      
    

    main_s1_set_seed(hp)
    main_s2_start_worker(main_worker, args, hp)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    args.hp = get_hyperparam(args)
    if args.distributed:
        if args.hp.multi_gpu.dist_url == "env://" and args.hp.multi_gpu.rank == -1:
            args.hp.multi_gpu.rank = int(os.environ["RANK"])
        if args.hp.multi_gpu.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.hp.multi_gpu.rank = args.hp.multi_gpu.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.hp.multi_gpu.dist_backend, init_method=args.hp.multi_gpu.dist_url,
                                world_size=args.world_size, rank=args.hp.multi_gpu.rank)
    # create model
    if args.hp.pretrained:
        print("=> using pre-trained model '{}'".format(args.hp.arch))
    else:
        print("=> creating model '{}'".format(args.hp.arch))
    if args.hp.model_source == eppb.HyperParam.ModelSource.TorchVision:
        model = torchvision.models.__dict__[args.hp.arch](pretrained=args.hp.pretrained)
    elif args.hp.model_source == eppb.HyperParam.ModelSource.PyTorchCV:
        model = ptcv_get_model(args.hp.arch, pretrained=args.hp.pretrained)
    elif args.hp.model_source == eppb.HyperParam.ModelSource.Local:
        model = imagenet_extra_models.__dict__[args.hp.arch](pretrained=args.hp.pretrained)


    print('model:\n=========\n')

    process_model(model, args, replace_first_layer=True, replace_map={
        'Conv2d': [my_nn.Conv2dLSQ],
        'Linear': [my_nn.LinearLSQ],
    }, nbits_w=args.hp.nbits_w, nbits_a=args.hp.nbits_a, qmode = args.hp.qmode)

    display_model(model)
    
    teacher_model = torchvision.models.__dict__["resnet101"](pretrained=args.hp.pretrained)
    
    # parallel and multi-gpu
    model = distributed_model(model, ngpus_per_node, args)
    teacher_model = distributed_model_teacher(teacher_model, ngpus_per_node, args)
    
    
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    cudnn.benchmark = True

    df = DataloaderFactory(args)
    train_loader, val_loader, train_sampler = df.product_train_val_loader(df.imagenet2012) 
    
    
    writer = get_summary_writer(args, ngpus_per_node, model)
    
    
    args.batch_num = len(train_loader)
    if args.hp.evaluate:
        
        exit_positions = torch.zeros(17)
        exit_positions[-1] = 1
        for name, m in model.named_modules():
            # print(name)
            if('conv1' in name and 'layer' in name and "exit" in name):
                layer_id = int(name[12])
                block_id = int(name[14])
                layer_num = 1+((layer_id-1)*4) + block_id * 2 + 1
                print("Exit present after Layer number: ", layer_num)
                exit_positions[layer_num] = 1
            
        num_exits = int(exit_positions.sum())
        if args.hp.ensemble:
            ensemble_model = nn.ModuleList()
            for i in range(1,num_exits):
                ensemble_model.append(imagenet_extra_models.ensemble.GeometricWeightedEnsemble(i+1,1000, model,i))
            checkpoint = torch.load(args.hp.resume_ensemble, map_location='cpu')
            
            new_checkpoint = copy.deepcopy(checkpoint)
            for key in new_checkpoint['state_dict'].keys():
                if key in checkpoint['state_dict'].keys():
                    temp_ = checkpoint['state_dict'].pop(key)
                    # key = key.replace('module.', '')
                    checkpoint['state_dict'][key] = temp_
            ensemble_model.load_state_dict(checkpoint['state_dict'])
            ensemble_model.cuda()
        else:
            ensemble_model = None

        if ensemble_model is not None:
            acc1, acc5 = validate(val_loader, ensemble_model[-1], criterion, args)
        else:
            acc1, acc5 = validate(val_loader, model, criterion, args)


        confidence = torch.tensor([0.80, 0.78])
        patience = 1
        
        
        
        for i in range(confidence.shape[0]):
            acc1, acc5, acc1_exit, best_samples_exit1, exits, incorrect_exits = validate_ee_patience_v2(val_loader, model, ensemble_model, criterion, args, confidence[i],patience, -1, num_exits)
            best_samples_exit2 = 0
            print("CONFIDENCE2 : "+str(confidence[i].item()) + " PATIENCE : " + str(patience))
            
            bitops_layer = torch.zeros(17)
            for name, m in model.named_modules():
                if ('conv1' in name and 'layer' not in name and 'exit' not in name):
                    bitops_layer[0] =  (torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops).item()
            for name, m in model.named_modules():
                if('conv1' in name and 'layer' in name and 'exit' not in name):
                    layer_id = int(name[12])
                    block_id = int(name[14])
                    layer_num = 1+((layer_id-1)*4) + block_id * 2 
                    bitops_layer[layer_num] += (torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops).item()
                elif('conv2' in name and 'layer' in name and 'exit' not in name):
                    layer_id = int(name[12])
                    block_id = int(name[14])
                    layer_num = 1+((layer_id-1)*4) + block_id * 2 + 1
                    bitops_layer[layer_num] += (torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops).item()
                elif('downsample.0' in name and 'layer' in name and 'exit' not in name):
                    layer_id = int(name[12])
                    block_id = int(name[14])
                    layer_num = 1+((layer_id-1)*4) + block_id * 2 + 1
                    bitops_layer[layer_num] += (torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops).item()
                elif('fc' in name and 'layer' not in name):
                    bitops_layer[-1] += (torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops).item()
                elif(('exit' in name and 'layer' in name) and ('conv' in name or 'fc' in name)):
                    layer_id = int(name[12])
                    block_id = int(name[14])
                    layer_num = 1+((layer_id-1)*4) + block_id * 2 + 1
                    bitops_layer[layer_num] += (torch.round(m.bits_a) * torch.round(m.bits_w) * m.bitops).item()
            bitops_layer = bitops_layer / 1000000000
            bitops_total = bitops_layer.sum()            
            bitops_exit = torch.zeros(num_exits)
            j = 0
            for k in range(exit_positions.shape[0]):
                if(exit_positions[k] == 1):
                    bitops_exit[j] = (bitops_layer[0:(k+1)]).sum()
                    j +=1
            print("Bitops Exit: ", bitops_exit)
            print("Exits: ", exits)
            print("Incorrect exits: ", incorrect_exits)
            bitops_total = (bitops_exit * exits/100).sum()
            print("Bitops Total Early Exit: ", bitops_total)
        return

    # for epoch in range(0, args.hp.start_epoch):
    #     scheduler_lr.step()

    global best_bits_weight
    global best_bits_activation
    global best_bitops 
    global bit_train_flag 
    global switch   

    best_bits_weight = 0
    best_bits_activation = 0
    best_bitops = 0
    bit_train_flag = 1
    switch = 0
    target_bitops = 20
    epoch1 = args.hp.epoch_stage1 #30  ## Pretraining exits stage
    epoch2 = args.hp.epoch_stage2 #60  ## Bit search stage
    epoch3 = args.hp.epoch_stage3 #30  ## Finetuning with searched precision
    if args.hp.ensemble:
        epoch4 = args.hp.epoch_stage4 #10  ## Ensemble stage
    else:
        epoch4 = 0
    
    
    stage = 1
    start_epoch = args.hp.start_epoch
    if (start_epoch < epoch1):
        stage = 1
    elif (start_epoch < epoch1+epoch2):
        stage = 2
    elif (start_epoch < epoch1+epoch2+epoch3):
        stage = 3
    elif (start_epoch < epoch1+epoch2+epoch3+epoch4):
        stage = 4
    
    
    exit_positions = torch.zeros(17)
    exit_positions[-1] = 1
    for name, m in model.named_modules():
        if('conv1' in name and 'layer' in name and "exit" in name):
            layer_id = int(name[12])
            block_id = int(name[14])
            layer_num = 1+((layer_id-1)*4) + block_id * 2 + 1
            exit_positions[layer_num] = 1

    
        
    num_exits = int(exit_positions.sum())
    args.hp.lr = args.hp.lr_stage1
    args.hp.epochs = epoch1
    optimizer = get_optimizer(model, args)
    scheduler_lr = get_lr_scheduler(optimizer, args)
    
    #####Stage 1 Training##########
    if (stage == 1):
        print("###########STARTING STAGE 1##############")
        for epoch in range(start_epoch, epoch1):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            for name, param in model.named_parameters():
                if("bits" in name):
                    param.requires_grad = False
            print("learning rate: ", scheduler_lr.get_last_lr())

            train_resnet18_first(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer)
            if (epoch % 10 == 9):
                for idx in range(num_exits):
                    print("Exit ", idx)
                    acc1, acc5, acc1_exit, best_samples_exit1 = validate_ee(val_loader, model, criterion, args, torch.tensor(0.9), idx)
                    if writer is not None:
                        writer.add_scalar('val/acc1_exit'+str(idx), acc1, epoch)
                        writer.add_scalar('val/acc5_exit'+str(idx), acc5, epoch)

            acc1, acc5 = validate(val_loader, model, criterion, args)
            if writer is not None:
                writer.add_scalar('val/acc1', acc1, epoch)
                writer.add_scalar('val/acc5', acc5, epoch)
                writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], epoch)
            
            scheduler_lr.step()
            
            is_best = acc1 > best_acc1
            if(is_best):
                best_acc1 = acc1
            
            
            if writer is not None:
                writer.add_scalar('val/best_acc1', best_acc1, epoch)
                writer.add_scalar('bits/best_bits_weight', best_bits_weight, epoch)
                writer.add_scalar('bits/best_bits_activation', best_bits_activation, epoch)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, prefix='{}/{}'.format(args.log_name, args.arch), epoch = epoch)
            if (args.hp.multi_gpu.rank == 0) :
                print("Best accuracy till now: "  + str(best_acc1.item()))
        stage = 2
        start_epoch = epoch1

    
    args.hp.epochs = epoch2
    args.hp.lr = args.hp.lr_stage2
    optimizer = get_optimizer(model, args)
    scheduler_lr = get_lr_scheduler(optimizer, args)
    for _ in range(epoch1, start_epoch):
        scheduler_lr.step()
    for name, param in model.named_parameters():
        if ("bits" in name):
            param.requires_grad = True
                    
    #####Stage 2 Training##########  
    if (stage == 2):
        print("###########STARTING STAGE 2##############")
        for epoch in range(start_epoch, epoch1+epoch2):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            print("learning rate: ", scheduler_lr.get_last_lr())
            bits_per_weight, bits_per_activation, bitops = train_resnet18(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer, target_bitops, bit_train_flag, switch)
            
            # evaluate on validation set
            acc1, acc5 = validate(val_loader, model, criterion, args)
            if writer is not None:
                writer.add_scalar('val/acc1', acc1, epoch)
                writer.add_scalar('val/acc5', acc5, epoch)
                writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], epoch)
            if (epoch % 10 == 9):
                for idx in range(num_exits):
                    print("Exit ", idx)
                    acc1, acc5, acc1_exit, best_samples_exit1 = validate_ee(val_loader, model, criterion, args, torch.tensor(0.9), idx)
                    if writer is not None:
                        writer.add_scalar('val/acc1_exit'+str(idx), acc1, epoch)
                        writer.add_scalar('val/acc5_exit'+str(idx), acc5, epoch)
            scheduler_lr.step()
            
            # # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            if(is_best):
                best_bits_weight = bits_per_weight
                best_bits_activation = bits_per_activation
                best_bitops = bitops
                best_acc1 = acc1
            if (args.hp.multi_gpu.rank == 0) :
                print("Bits per weight : " + str((bits_per_weight).item()))
                print("Bits per activation : " + str((bits_per_activation).item()))
                print("Bitops : " + str(bitops.item()))
                print("Best accuracy till now: "  + str(best_acc1.item()) + " @ " + str(best_bits_weight.item()) + " bits per weight, " + str(best_bits_activation.item()) + " bits per activation, "+ str(best_bitops.item()) + " G bitops")
            
            if writer is not None:
                writer.add_scalar('val/best_acc1', best_acc1, epoch)
                writer.add_scalar('bits/best_bits_weight', best_bits_weight, epoch)
                writer.add_scalar('bits/best_bits_activation', best_bits_activation, epoch)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, prefix='{}/{}'.format(args.log_name, args.arch), epoch = epoch)
            
            if epoch % 10 == 0:
                for idx in range(num_exits):
                    print("Exit ", idx)
                    acc1, acc5, acc1_exit, best_samples_exit1 = validate_ee(val_loader, model, criterion, args, torch.tensor(0.9), idx)
        
        
        for idx in range(num_exits):
            print("Exit ", idx)
            acc1, acc5, acc1_exit, best_samples_exit1 = validate_ee(val_loader, model, criterion, args, torch.tensor(0.9), idx)
            if writer is not None:
                    writer.add_scalar('val/acc1_exit'+str(idx), acc1, epoch)
                    writer.add_scalar('val/acc5_exit'+str(idx), acc5, epoch)
        
        stage = 3
        start_epoch = epoch1+epoch2

    
    args.hp.epochs = epoch3
    args.hp.lr = args.hp.lr_stage3
    optimizer = get_optimizer(model, args)
    scheduler_lr = get_lr_scheduler(optimizer, args)
    for _ in range(epoch1+epoch2, start_epoch):
        scheduler_lr.step()
    
    #####Stage 3 Training##########  
    if (stage == 3):
        print("###########STARTING STAGE 3##############")
        for epoch in range(start_epoch, epoch1+epoch2+epoch3):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            
            #reset best acc after stage 2 and set bits to non trainable and round them
            if (epoch == start_epoch):
                bit_train_flag = 0
                best_acc1 = 0
                switch = 0
                for name, m in model.named_parameters():
                    if('bits' in name):
                        bits = m.data.clone()
                        bits = torch.round(bits)
                        m.data.copy_(bits)
                        m.requires_grad = False
            # teacher_model = None
            print("learning rate: ", scheduler_lr.get_last_lr())
            bits_per_weight, bits_per_activation, bitops = train_resnet18(train_loader, model, teacher_model, criterion, optimizer, epoch, args, writer, target_bitops, bit_train_flag, switch)

            # evaluate on validation set
            acc1, acc5 = validate(val_loader, model, criterion, args)
            if writer is not None:
                writer.add_scalar('val/acc1', acc1, epoch)
                writer.add_scalar('val/acc5', acc5, epoch)
                writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], epoch)
            
            scheduler_lr.step()
            
            # # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            if(is_best):
                best_bits_weight = bits_per_weight
                best_bits_activation = bits_per_activation
                best_bitops = bitops
                best_acc1 = acc1
            if (args.hp.multi_gpu.rank == 0) :
                print("Bits per weight : " + str((bits_per_weight).item()))
                print("Bits per activation : " + str((bits_per_activation).item()))
                print("Bitops : " + str(bitops.item()))
                print("Best accuracy till now: "  + str(best_acc1.item()) + " @ " + str(best_bits_weight.item()) + " bits per weight, " + str(best_bits_activation.item()) + " bits per activation, "+ str(best_bitops.item()) + " G bitops")
            
            if writer is not None:
                writer.add_scalar('val/best_acc1', best_acc1, epoch)
                writer.add_scalar('bits/best_bits_weight', best_bits_weight, epoch)
                writer.add_scalar('bits/best_bits_activation', best_bits_activation, epoch)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, prefix='{}/{}'.format(args.log_name, args.arch), epoch = epoch)
            
            if epoch % 10 == 0:
                for idx in range(num_exits):
                    print("Exit ", idx)
                    acc1, acc5, acc1_exit, best_samples_exit1 = validate_ee(val_loader, model, criterion, args, torch.tensor(0.9), idx)
    
        for idx in range(num_exits):
            print("Exit ", idx)
            acc1, acc5, acc1_exit, best_samples_exit1 = validate_ee(val_loader, model, criterion, args, torch.tensor(0.9), idx)
    
        stage = 4
        start_epoch = epoch1+epoch2+epoch3

    
    for _ in range(epoch1+epoch2+epoch3, start_epoch):
        scheduler_lr.step()
    
    
    # #freeze original model
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    #create ensemble model
    ensemble_model = nn.ModuleList()
    for i in range(1,num_exits):
        ensemble_model.append(imagenet_extra_models.ensemble.GeometricWeightedEnsemble(i+1,1000, model, i))
    ensemble_model = ensemble_model.cuda()
    torch.cuda.set_device(args.hp.multi_gpu.rank) # my local gpu_id
    
    
    args.hp.epochs = epoch4
    args.hp.lr = args.hp.lr_stage4
    optimizer = get_optimizer(ensemble_model, args)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=11, gamma=0.1)
    criterion = torch.nn.NLLLoss().cuda(args.gpu)

    if (stage == 4):
        print("###########STARTING STAGE 4##############")
        # acc1, acc5 = validate(val_loader, model, criterion, args)
        for exit_id in range(num_exits-2,num_exits-1):
            print("###########EXIT ", exit_id+2, " ###########")
            # scheduler_lr = get_lr_scheduler(optimizer, args)
            for epoch in range(start_epoch, epoch1+epoch2+epoch3+epoch4):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                #reset best acc after stage 3 
                if (epoch == epoch1+epoch2+epoch3):
                    best_acc1 = torch.tensor([0]).cuda()

                # for name, m in ensemble_model[exit_].named_modules():
                print(ensemble_model[exit_id]._weight)
                train_ensemble_resnet18(train_loader, ensemble_model[exit_id].base_model, ensemble_model[exit_id], None, criterion, optimizer, epoch, args, writer, target_bitops, bit_train_flag, switch, exit_id+1)
                
                # evaluate on validation set
                acc1, acc5, _ = validate_ee_ensembles(val_loader, model, ensemble_model[exit_id], criterion, args, idx = exit_id+1)
                if writer is not None:
                    writer.add_scalar('val/acc1', acc1, epoch)
                    writer.add_scalar('val/acc5', acc5, epoch)
                    writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], epoch)
                
                scheduler_lr.step()
                
                # # remember best acc@1 and save checkpoint
                max_accuracy_across_devices = max(sync_tensor_across_gpus(acc1))
                is_max_device = max_accuracy_across_devices == acc1
                is_best = (max_accuracy_across_devices > best_acc1)
                if(is_best):
                    best_acc1 = max_accuracy_across_devices
                    if (is_max_device):
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': ensemble_model.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer': optimizer.state_dict(),
                        }, is_best, prefix='{}/{}'.format(args.log_name, "ENSEMBLE"), epoch = epoch)
                print("Best accuracy till now: " , best_acc1)
                

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    
    main()
