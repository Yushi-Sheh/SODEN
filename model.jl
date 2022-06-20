using Flux

function make_net(input_size, hidden_size, num_layers, output_size, dropout=0,
        batch_norm=False, act="relu", softplus=True)
    if act== "selu"
        ActFn=selu
    else
        ActFn=relu
    end


end



function NonCoxFuncModel()
