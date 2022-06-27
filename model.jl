using Flux
using Base: @kwdef
using CUDA 
using Random


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
#struct 
@kwdef mutable struct BaseSurvODEFunc
    nfe::Int64=0
    batch_time_mode::Bool=false
end

@kwdef mutable struct ExpODEFunc
    nfe::Int64=0
    batch_time_mode::Bool=false
    #special
    lambda::Float64=0.1
end 

@kwdef mutable struct MLPODEFunc
    nfe::Int64=0
    batch_time_mode::Bool=false
    #special
    hidden_size::Int64
    num_layers::Int64
    batch_norm::Int64
    net
end

@kwdef mutable struct ContextRecMLPODEFunc
    nfe::Int64=0
    batch_time_mode::Bool=false
    #special
    feature_size::Int64
    hidden_size::Int64
    num_layers::Int64
    batch_norm::Int64
    use_embed::Int64
    embed 
    net 
end    

ODEFuncType=Union{BaseSurvODEFunc,ExpODEFunc,MLPODEFunc,ContextRecMLPODEFunc}

#function
function set_batch_time_mode(obj::ODEFuncType,mode::Bool)
    obj.batch_time_mode=mode
end

function reset_nfe(obj::ODEFuncType)
    obj.nfe=0
end

function ExpODEFunc_forward(obj::ExpODEFunc,t,y)
    obj.nfe+=1
    if obj.batch_time_mode
        return obj.lambda*ones(size(t))
    else
        return obj.lambda
    end
end

function MLPODEFunc_forward(obj::MLPODEFunc,t,y)
    obj.nfe+=1
    #not implement device part
    #T = y.index_select(-1, torch.tensor([1]).to(device)).view(-1, 1)
    #inp = t.repeat(T.size()) * T
    #output = self.net(inp) * T
    zero_m=zeros(size(T))
    #in pytorch is 1
    output=cat(output,zero_m,dims=2)
    if obj.batch_time_mode
        return output
    else
        return squeeze(1,output)
    end
end

function ContextRecMLPODEFunc_forward(obj::ContextRecMLPODEFunc,t,y)
    obj.nfe+=1
    #device = next(self.parameters()).device
end

function make_rnn(RNNModel,input_size,hidden_size,num_layers)
    layers=Any[RNNModel(input_size=>hidden_size)]
    for i in 2:num_layers
        push!(layers,RNNModel(hidden_size=>hidden_size))
    end
    return Chain(layers...)
end

@kwdef mutable struct CoxFuncModel
    model_config=nothing
    feature_size =nothing
    func_type =nothing
    has_feature =nothing
    use_embed =nothing
    odefunc=nothing
    beta=nothing
    x_net=nothing
    embed=nothing 
    last_eval=nothing 
end

function CoxFuncModel_init(obj::CoxFuncModel,model_config,feature_size,use_embed)

end 

mutable struct SODEN 
    rnn_config 
    seq_feat_size 
    rnn 
    feature_size 
    model 
end


function SODEN_init(obj::SODEN,model_config,feature_size,use_embed)
    #model_config: dict 
    rnnParam=get(model_config,"rnn",-1)
    if rnnParam!=-1
        obj.rnn_config=get(rnnParam,"rnn_0",-1)
        if get(obj.rnn_config,"rnn_type",-1)=="LSTM"
            RNNModel=LSTM
        elseif get(obj.rnn_config,"rnn_type",-1)=="GRU"
            RNNModel=GRU
        else 
            error("Unsupported RNN type: ",get(obj.rnn_config,"rnn_type",-1))
        end
        obj.seq_feat_size=get(feature_size,"seq_feat",-1)
        obj.rnn=make_rnn(RNNModel,obj.seq_feat_size,get(obj.rnn_config,"hidden_size",-1),get(obj.rnn_config,"num_layers",-1))
        feature_size=get(obj.rnn_config,"hidden_size",-1)+get(feature_size,"fix_feat",-1)
    else 
        obj.rnn=nothing
    end
    config=get(get(model_config,"ode",-1),"surv_ode_0",-1)
    if get(config,"layer_type",-1)=="surv_ode"
        if get(config,"func_type",-1) in ["rec_mlp"]
            #not initialized
            obj.model=NonCoxFuncModel()
        elseif get(config,"func_type",-1) in ["cox_mlp_exp","cox_mlp_mlp"]
            #not initialized
            obj.model=CoxFuncModel()
        else
            error("func_type ",get(config,"func_type",-1)," not supported.")
        end 
    else 
        error("Model ",get(config,"layer_type",-1)," not supported.")
    end
end

function SODEN_set_last_eval(obj::SODEN,last_eval=true)
    if "set_last_eval" in keys(obj.model)
        set_last_eval(last_eval)
    end 
end

function SODEN_forward(obj::SODEN,inputs)
    #what is inputs?
    if isnothing(obj.rnn) ==false

    end
end
#define dict_type
#get(collection, key, default) Return the value stored 
#for the given key, or the given default value if no mapping for the key is present.

#m=make_net(10,12,3,12,0.1,true,"relu",true)
#print(make_net(10,12,3,12,0.1,true,"relu",true))
#x=randn(10,2)
#print(m(x))
#function NonCoxFuncModel()
