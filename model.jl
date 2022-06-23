using Flux

function make_net(input_size, hidden_size, num_layers, output_size, dropout=0,
        batch_norm=false, act="relu", softplus=True)
    if act== "selu"
        ActFn=selu
    else
        ActFn=relu
    end
    modules=Any[Dense(input_size,hidden_size,ActFn)]
    if batch_norm
        push!(modules,BatchNorm(hidden_size))
    end
    if dropout>0
        push!(modules, Dropout(dropout))
    end
    if num_layers>1
        for i in 1:num_layers-1
            push!(modules,Dense(hidden_size,hidden_size,ActFn))
            if batch_norm
                push!(modules,BatchNorm(hidden_size))
            end
            if dropout>0
                push!(modules,Dropout(dropout))
            end
        end
    end
    push!(modules, Dense(hidden_size,output_size))
    push!(modules, softmax)
    return Chain(modules...)
end

#m=make_net(10,12,3,12,0.1,true,"relu",true)
#print(make_net(10,12,3,12,0.1,true,"relu",true))
#x=randn(10,2)
#print(m(x))
#function NonCoxFuncModel()
